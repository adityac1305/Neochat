# test_connection.py
from graph import graph

def test_connection():
    try:
        result = graph.query("RETURN 1 AS ok")
        print("Neo4j connection successful:", result)
    except Exception as e:
        print("Neo4j connection failed:", e)

if __name__ == "__main__":
    test_connection()
    result = graph.query("MATCH (n) RETURN n LIMIT 5")
    print(result)