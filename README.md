# RAG with additional information about the quality of embeddings
## External links
Dataset: [Trivia QA](https://www.codecademy.com/resources/docs/markdown/links)\
Report: [GoogleSlides](https://docs.google.com/presentation/d/1YtCDubkZvWOlEp85khHYN8fSJMHsnkWiHCWjvJKTDJ8/edit#slide=id.g2f1a2caa45c_0_19)

## Architecture
![architecture](https://github.com/Pomelkin/smiles-rag/blob/main/images/architecture.jpg)
### Overview
This architecture is designed to optimize the process of generating accurate and contextually appropriate responses to user queries. The solution leverages advanced clustering, vector retrieval, and model inference techniques to ensure high-quality outputs.
### Workflow
1. **User Query**\
    A user submits a query, which is then vectorized using an embedding model.
2. **Vector Retrieval with Qdrant**\
    The vectorized query is sent to a Qdrant database.
    Qdrant uses cosine similarity to identify and retrieve the 9 most similar vectors from the database.
3. **Vector Clustering**\
    The 9 retrieved vectors are grouped into 3 clusters using the k-means clustering algorithm.
4. **Vector Selection**\
    From the 9 vectors, the top-1 vector based on cosine similarity is selected and its cluster is fixed.
    From the remaining 2 clusters, one random vector is selected from each, resulting in 3 vectors from distinct clusters.
5. **Intermediate Processing (Gemma 2 2B)**\
    Each selected vector is processed by an intermediate expert model, Gemma 2 2B.
    The model generates individual responses based on each vector, producing 3 distinct responses corresponding to different aspects of the query.
6. **Metric Calculation**\
    A preference metric is calculated for the top-1 vector.
    An uncertainty metric is calculated for all three vectors.
7. **Final Response Generation (LLaMA 3.1B)**\
    The preliminary responses and their associated metrics are passed to the final model, LLaMA 3.1B.
    LLaMA 3.1B evaluates the responses and metrics to generate the final answer to the user’s query.
### Key Advantages
1. Diversified Vector Clustering: Clustering ensures that the retrieved vectors are diverse, preventing the concentration of information around a single topic or aspect of the query.
2. Thorough Intermediate Processing: The use of intermediate models allows each vector to be thoroughly processed, improving the overall quality of the final response.
3. Context-Aware Final Response: By integrating metrics into the response generation process, the final model (LLaMA 3.1B) is able to consider context more accurately, resulting in a more precise and relevant answer.
