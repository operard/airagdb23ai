
import oracledb
import array

import configparser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores import OracleVS
from langchain_community.document_loaders import oracleai
from langchain_community.utilities.oracleai import OracleSummary
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.document_loaders.oracleai import OracleTextSplitter 
from langchain_community.document_loaders.oracleai import OracleDocLoader 

from langchain_huggingface import HuggingFaceEmbeddings


# ======== parameter 

config = configparser.ConfigParser()
config.read("/config/config.env")
#config.read("/home/cloud-user/database/config.env")

username = config["DATABASE"]["USERNAME"]
password = config["DATABASE"]["PASSWORD"]
host = config["DATABASE"]["HOST"]
port = config["DATABASE"]["PORT"]
service_name = config["DATABASE"]["SERVICE_NAME"]
table_name = config["DATABASE"]["TABLE_NAME"]

dsn=host+":"+port+"/"+service_name

distance_function = 'DOT'

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5" # Uses 

# ============== Functions ====================


def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Create text chunks from a large text block."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Function to check access to DB23ai
def checkDatabaseAccess (tablename):
    try:
        print("connecting...  to Oracle Database to checkDatabaseAccess")
        
        # Thick Driver in Python for Vector
        oracledb.init_oracle_client()
    
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        
        print("Successfully connected to Oracle Database")
        
        cursor = connection.cursor()
        
        # Create Collection : Chunk Size = 1000 and embeddings size = 1024
        cursor.execute("""
            create table if not exists {} (
                doc_id number(5),
                filepath varchar2(2000),
                file_name varchar2(1000),
                model_name varchar2(100),
                chunk_id number(5),
                embed_data varchar2(4000),
                embed_vector vector(1024, float32),
                primary key (doc_id))""".format(table_name))
            
        # Close Connection
        connection.close()
        
        return 0
    except Exception as error:
        print(f"--> checkDatabaseAccess, Error ocurred : {error}",error)
        return -1        


# Function to get select  MAX(DOC_ID) FROM AIRAGINBOX;
def getMaxID(tablename, cursor):
    cursor.execute("select  MAX(DOC_ID) AS ID FROM {}".format(tablename))
    while True:
        row = cursor.fetchone()
        if row is None:
            return 0
        else:
            return int(row[0])


# Function to insert a ROW in a table in DB23ai
def insertRow (tablename, cursor, id_val, v_filepath, v_filename, v_model_name, chunk_id, text, embeddings):
    # print (embeddings)
    input_array = array.array("f", embeddings)
    # Insert Points
    cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR)
    cursor.execute("insert into {} values (:1, :2, :3, :4, :5, :6, :7)".format(tablename), [id_val, 
                                            str(v_filepath), 
                                            str(v_filename), 
                                            str(v_model_name), 
                                            chunk_id, 
                                            text,
                                            input_array])


# Function to store File in DB23ai
def storeChunks (v_filename, chunks):
    print("connecting...  to Oracle Database to insert Chunks: "+len(chunks))
    print(username)
    
    # Thick Driver in Python for Vector
    oracledb.init_oracle_client()

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    
    print("Successfully connected to Oracle Database")
    
    cursor = connection.cursor()
    

    # Get MAX DOC_ID
    docid = getMaxID(table_name, cursor)
    
    # Dimensions : 1024
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
        
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        # Insert row    
        insertRow (table_name, cursor, docid, v_filename, v_filename, EMBEDDING_MODEL_NAME, i, chunk, vector)
        # Next ID
        docid += 1
        
    # Close Connection
    connection.close()
    

# Function to execute RAG Search in DB23ai
def rag_search (prompt_question_corret, k_top):
    print("connecting...  to Oracle Database to prepare RAG")
    print(username)
    
    # Thick Driver in Python for Vector
    oracledb.init_oracle_client()

    connection = oracledb.connect(user=username, password=password, dsn=dsn)

    print("connected to Oracle Database")
    
    # Locally stored Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)    
    
    # Connect to vector DB
    vector_store_dot = OracleVS(
        client=connection,
        embedding_function=embeddings,
        table_name=table_name,
        distance_strategy=distance_function,
    )

    # Upload to texts
    # vector_store_dot.add_texts(list_chunks)

    # Retrieve Answers
    answers = vector_store_dot.similarity_search (prompt_question_corret,  k_top)    
    
    # Close connection
    connection.close()
    
    return answers



# Function implemented by Luca Bindi
def rag_search_old (prompt_question_corret, k_top):
    print("connecting...  to Oracle Database")
    print(username)
    
    # Thick Driver in Python for Vector
    oracledb.init_oracle_client()

    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    print("connected to Oracle Database")
    #query to find the closest vector embeding related input question vector
    sql1 = """select  TO_CHAR(vector_distance(EMBEDDING,(select json_value(COLUMN_VALUE, '$.embed_vector' RETURNING clob) 
                                                              from dbms_vector.UTL_TO_EMBEDDINGS(:prompt_question_corret,
                                                            json('{"provider":"database", "model": "gistembedding"}'))
                                                            ),"""+distance_function+""" 
                                              ),'9.99999999999999999999999999999999999999') Distance
               ,text|| ' summary :'||json_value(METADATA,'$.document_summary'),
                substr(json_value(METADATA,'$._file'),instr(json_value(METADATA,'$._file'),'/',-1))     
  from """ +table_name+""" where 
  vector_distance(EMBEDDING,(select json_value(COLUMN_VALUE, '$.embed_vector' RETURNING clob) 
from 
  dbms_vector.UTL_TO_EMBEDDINGS(:prompt_question_corret,
    json('{"provider":"database", "model": "gistembedding"}') )),"""+distance_function+""" )
order by vector_distance(EMBEDDING,(select json_value(COLUMN_VALUE, '$.embed_vector' RETURNING clob) 
from 
  dbms_vector.UTL_TO_EMBEDDINGS(:prompt_question_corret,
    json('{"provider":"database", "model": "gistembedding"}') )),"""+distance_function+""" ) 
FETCH FIRST :k_top ROWS ONLY
"""

    bind_values = {"prompt_question_corret": prompt_question_corret,"k_top":k_top}  # Map placeholders to values
    with connection.cursor() as cursor:
            cursor.execute(sql1,bind_values)
            rows = cursor.fetchall()

    # Generate the answer using chunk obtained from the database 
    input_sentence ='Answer this question : "' + prompt_question_corret  + '" based solely on this information  :  ' ;
    for i in range(len(rows)):
        input_sentence += str(rows[i][1])+' and  '
    File_rif='File  Ref : \n'
    seen_values = set()
    for row in rows:
        value = row[2]
        if value not in seen_values:
            File_rif += "\n --"+ value + '\n'
            seen_values.add(value)
    chunk_info= 'Chunk info : \n '
    for i in range(len(rows)):
        chunk_info += '\n Distance : '+rows[i][0]+ ' Chunk : '+str(rows[i][1])+' \n ************* \n'        
    print(input_sentence)
    print(File_rif)
    # Close connection
    connection.close()
    
    return input_sentence,File_rif,chunk_info
