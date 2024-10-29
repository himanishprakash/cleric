## Approach

1. **Setting Up Docker and Kubernetes**:  
   First, I set up Docker and Kubernetes to create a manageable, local environment. Minikube made this easy by allowing me to spin up a single-node Kubernetes cluster on my machine. This setup was essential to provide a foundation for the AI agent to interact with and retrieve information from.

2. **Deploying a Test Application**:  
   With Kubernetes running, I deployed a simple application (like NGINX) to test that everything was configured correctly. This also gave me a reference point to check if the AI agent could successfully access information about running applications.

3. **Extracting Data from Kubernetes**:  
   Once the cluster was working as expected, I moved on to developing the `DataExtractor` class. This part of the code connects to the Kubernetes API, fetching details about the cluster's resources, including pods, deployments, services, nodes, config maps, and more. Collecting this data into a structured format was essential for giving the AI a solid knowledge base to draw from.

4. **Building the AI Query Agent**:  
   Next, I integrated OpenAI’s language model to interpret and respond to natural language queries about the Kubernetes cluster. I spent some time fine-tuning the AI’s prompt to get accurate, straightforward answers. This was a bit of trial and error—adjusting the prompt until I found a balance that gave clear, helpful responses without extra fluff.

5. **Testing and Fine-Tuning**:  
   Finally, I set up a FastAPI server with a `/query` endpoint where users can ask questions about the cluster. By orchestrating the data extraction and AI model in tandem, the server processes each query, fetches up-to-date cluster info, and uses the AI to format an answer in simple language. I ran several tests to ensure the agent accurately understood and responded to different types of questions.

