from pymilvus import connections, db, MilvusClient

# conn = connections.connect(host="127.0.0.1", port=19530)


client = MilvusClient()

#%%
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
    auto_id=True
)


# db.create_collection("description_demo")


# database = db.create_database("book")
# db.drop_database("book")



#%%
from pymilvus import DataType, CollectionSchema, FieldSchema, Collection

# 定义集合和字段（如果尚未定义）
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
text_field = FieldSchema(name="thought", dtype=DataType.VARCHAR, max_length=1024)
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="Store thoughts and their embeddings")
collection = Collection(name="thoughts", schema=schema)

# 定义索引参数
index_params = {
    "index_type": "IVF_FLAT",  # 保持使用 IVF_FLAT
    "metric_type": "IP",       # 使用内积作为度量类型
    "params": {"nlist": 50}    # 对于较小的数据集，减少 nlist 的值
}


# 创建索引
collection.create_index(field_name="vector", index_params=index_params)
print("Index created successfully.")





#%%
from pymilvus import model


embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"vector": vectors[i], "thought": docs[i]}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))




#%%
client = MilvusClient()
res = client.insert(collection_name="thoughts", data=data)
print(res)

#%%
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
print(query_vectors)
res = client.search(
    collection_name="thoughts",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["thought"],  # specifies fields to be returned
)
print(res[0])


#%%
# 检查集合是否已加载，如果没有，则加载

client.load_collection("thoughts")


# 现在执行搜索
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
res = client.search(
    collection_name="thoughts",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["thought"],  # specifies fields to be returned
)

# 打印结果

print(res[0][0]["entity"]["thought"])


#%%
# print(res[0][1]["entity"]["thought"])
print(res[0][0]["entity"]["text"])


#%%
client.drop_collection("demo_collection")
#%%
client.drop_collection("thoughts")

#%%
from pymilvus import connections, db

conn = connections.connect(host="127.0.0.1", port=19530)

print(client.list_collections())
