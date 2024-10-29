from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kubernetes import client, config
import logging
import os
from typing import List, Dict, Any
import json
import OpenAI
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

class DataExtractor:
    """Agent responsible for extracting Kubernetes cluster information."""
    
    def __init__(self):
        """Initialize the Kubernetes clients."""
        try:
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            self.knowledge_base = {}
            self.last_update = None
            self.update_interval = 300  # 5 minutes
            logger.info("Successfully initialized DataExtractor")
        except Exception as e:
            logger.error(f"Failed to initialize DataExtractor: {str(e)}")
            raise

    def extract_kubernetes_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all relevant information from the Kubernetes cluster."""
        cluster_data = {
            "pods": [],
            "deployments": [],
            "services": [],
            "nodes": [],
            "configmaps": [],
            "ingresses": [],
            "persistent_volume_claims": []
        }

        try:
            # Pods
            try:
                pods = self.v1.list_pod_for_all_namespaces()
                for pod in pods.items:
                    cluster_data["pods"].append({
                        "name": pod.metadata.name,
                        "namespace": pod.metadata.namespace,
                        "status": pod.status.phase,
                        "node_name": pod.spec.node_name,
                        "containers": [container.name for container in pod.spec.containers]
                    })
            except Exception as e:
                logger.error(f"Error fetching pods: {str(e)}")
                cluster_data["pods"] = [{"error": f"Failed to fetch pods: {str(e)}"}]

            # Deployments
            try:
                deployments = self.apps_v1.list_deployment_for_all_namespaces()
                for deployment in deployments.items:
                    cluster_data["deployments"].append({
                        "name": deployment.metadata.name,
                        "namespace": deployment.metadata.namespace,
                        "replicas": deployment.spec.replicas,
                        "available_replicas": deployment.status.available_replicas
                    })
            except Exception as e:
                logger.error(f"Error fetching deployments: {str(e)}")
                cluster_data["deployments"] = [{"error": f"Failed to fetch deployments: {str(e)}"}]

            # Services
            try:
                services = self.v1.list_service_for_all_namespaces()
                for service in services.items:
                    cluster_data["services"].append({
                        "name": service.metadata.name,
                        "namespace": service.metadata.namespace,
                        "type": service.spec.type,
                        "cluster_ip": service.spec.cluster_ip
                    })
            except Exception as e:
                logger.error(f"Error fetching services: {str(e)}")
                cluster_data["services"] = [{"error": f"Failed to fetch services: {str(e)}"}]

            # Nodes
            try:
                nodes = self.v1.list_node()
                for node in nodes.items:
                    cluster_data["nodes"].append({
                        "name": node.metadata.name,
                        "status": [status.type for status in node.status.conditions if status.status == "True"],
                        "capacity": node.status.capacity
                    })
            except Exception as e:
                logger.error(f"Error fetching nodes: {str(e)}")
                cluster_data["nodes"] = [{"error": f"Failed to fetch nodes: {str(e)}"}]

            # ConfigMaps
            try:
                configmaps = self.v1.list_config_map_for_all_namespaces()
                for configmap in configmaps.items:
                    cluster_data["configmaps"].append({
                        "name": configmap.metadata.name,
                        "namespace": configmap.metadata.namespace
                    })
            except Exception as e:
                logger.error(f"Error fetching configmaps: {str(e)}")
                cluster_data["configmaps"] = [{"error": f"Failed to fetch configmaps: {str(e)}"}]

            # Ingresses
            try:
                ingresses = self.networking_v1.list_ingress_for_all_namespaces()
                for ingress in ingresses.items:
                    cluster_data["ingresses"].append({
                        "name": ingress.metadata.name,
                        "namespace": ingress.metadata.namespace,
                        "rules": [rule.host for rule in ingress.spec.rules] if ingress.spec.rules else []
                    })
            except Exception as e:
                logger.error(f"Error fetching ingresses: {str(e)}")
                cluster_data["ingresses"] = [{"error": f"Failed to fetch ingresses: {str(e)}"}]

            # Persistent Volume Claims
            try:
                pvcs = self.v1.list_persistent_volume_claim_for_all_namespaces()
                for pvc in pvcs.items:
                    storage = pvc.spec.resources.requests.get("storage", "N/A")
                    cluster_data["persistent_volume_claims"].append({
                        "name": pvc.metadata.name,
                        "namespace": pvc.metadata.namespace,
                        "storage": storage
                    })
            except Exception as e:
                logger.error(f"Error fetching PVCs: {str(e)}")
                cluster_data["persistent_volume_claims"] = [{"error": f"Failed to fetch PVCs: {str(e)}"}]

        except Exception as e:
            logger.error(f"Critical error in extract_kubernetes_data: {str(e)}")
            return {
                "error": f"Failed to extract Kubernetes data: {str(e)}",
                "pods": [],
                "deployments": [],
                "services": [],
                "nodes": [],
                "configmaps": [],
                "ingresses": [],
                "persistent_volume_claims": []
            }

        # Add metadata about the extraction
        cluster_data["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "errors": [
                resource for resource, data in cluster_data.items()
                if data and isinstance(data[0], dict) and "error" in data[0]
            ]
        }

        return cluster_data
    def update_knowledge_base(self):
        """Update the knowledge base with current cluster information."""
        try:
            current_time = datetime.now()
            if (self.last_update and 
                (current_time - self.last_update).total_seconds() < self.update_interval):
                return

            self.knowledge_base = {
                'data': self.extract_kubernetes_data(),
                'last_updated': current_time.isoformat()
            }
            
            self.last_update = current_time
            logger.info("Successfully updated knowledge base")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            raise



class QueryAnalyzer:
    """Agent responsible for analyzing queries and finding answers in the knowledge base."""
    
    def __init__(self):
        """Initialize the Query Analyzer."""
        try:
            # Create OpenAI client
            self.client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))  
            logger.info("Successfully initialized QueryAnalyzer")
        except Exception as e:
            logger.error(f"Failed to initialize QueryAnalyzer: {str(e)}")
            raise

    def clean_resource_names(self, answer: str) -> str:
        """Clean resource names in the answer."""
        # Remove random hash suffixes
        cleaned = re.sub(r'-[0-9a-f]{8,}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', '', answer)
        # Remove numeric suffixes
        cleaned = re.sub(r'-\d+$', '', cleaned)
        return cleaned

    def analyze_query(self, query: str, knowledge_base: Dict[str, Any]) -> str:
        """Analyze the query and find the answer in the knowledge base."""
        try:
            system_prompt = """
            You are a Kubernetes query analyzer. Given a knowledge base of cluster information, 
            provide ONLY the exact answer without any additional text. For example:
            - For count queries: return just the number (e.g., "2")
            - For name queries: return just the name (e.g., "nginx")
            - For status queries: return just the status (e.g., "Running")
            
            Rules:
            1. Return ONLY the answer value, no additional words or context
            2. Remove any random identifiers from resource names (e.g., "mongodb" not "mongodb-56c598c8fc")
            3. For counts, return just the number
            4. For status, return just the status word
            5. For names, return just the name
            6. Never add explanatory text
            7. Never add units or labels
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Knowledge base: {json.dumps(knowledge_base)}\nQuery: {query}"}
                ],
                temperature=0.1
            )

            answer = response.choices[0].message.content.strip()
            # Clean up the answer further
            answer = self.clean_resource_names(answer)
            # Remove any trailing periods or spaces
            answer = answer.rstrip('.')
            # Remove any "Answer:" or similar prefixes
            answer = re.sub(r'^(Answer|Result|Count|Status|Name):\s*', '', answer, flags=re.IGNORECASE)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return f"Error analyzing query: {str(e)}"
            
class QueryOrchestrator:
    """Orchestrates the interaction between the data extractor and query analyzer."""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.analyzer = QueryAnalyzer()
        logger.info("Query Orchestrator initialized")

    async def process_query(self, query: str) -> str:
        """Process a query through both agents and return the final response."""
        try:
            # Update knowledge base
            self.extractor.update_knowledge_base()
            
            # Analyze query using the knowledge base
            answer = self.analyzer.analyze_query(query, self.extractor.knowledge_base)
            
            return answer
        except Exception as e:
            logger.error(f"Error in query orchestration: {str(e)}")
            return "Error processing query"

# Initialize the orchestrator
orchestrator = QueryOrchestrator()

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle incoming queries and return responses."""
    try:
        logger.info(f"Received query: {request.query}")
        answer = await orchestrator.process_query(request.query)
        logger.info(f"Final answer: {answer}")
        return QueryResponse(query=request.query, answer=answer)
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)