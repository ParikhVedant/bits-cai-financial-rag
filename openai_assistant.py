import openai
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class OpenAIAssistant:
    def __init__(self, assistant_id="asst_UDHCxcwxtjw6H65JV90RKSEo"):
        """
        Initialize the OpenAI Assistant.
        
        Args:
            assistant_id: ID of the pre-created OpenAI Assistant
        """
        self.assistant_id = assistant_id
        api_key = os.getenv("OPENAI_API_KEY")
        
        print(f"API Key found: {api_key is not None}, First 10 chars: {api_key[:10] if api_key else 'None'}")
        print(f"Using OpenAI version: {openai.__version__}")
        
        # Initialize client with only API key, no org ID
        self.client = openai.OpenAI(api_key=api_key)
        self.thread_id = None
        
    def create_thread(self):
        """Create a new thread for the conversation."""
        thread = self.client.beta.threads.create()
        self.thread_id = thread.id
        return thread.id
    
    def get_thread_id(self):
        """Get current thread ID or create a new one if none exists."""
        if not self.thread_id:
            return self.create_thread()
        return self.thread_id
    
    def process_query(
        self, 
        query: str, 
        retrieval_method: str = None, 
        k: int = None
    ) -> Dict[str, Any]:
        """
        Process a user query using the OpenAI Assistant.
        
        Args:
            query: The user's question
            retrieval_method: Optional retrieval method to pass as metadata
            k: Optional number of chunks to retrieve
            
        Returns:
            Dict containing the response from the assistant
        """
        try:
            # Ensure we have a thread
            thread_id = self.get_thread_id()
            
            # Create message
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query
            )
            
            # Create run with additional instructions if provided
            additional_instructions = ""
            if retrieval_method and k:
                additional_instructions = f"Use {retrieval_method} retrieval method and consider top {k} chunks."
            
            run_params = {
                "thread_id": thread_id,
                "assistant_id": self.assistant_id
            }
            
            if additional_instructions:
                run_params["instructions"] = additional_instructions
            
            run = self.client.beta.threads.runs.create(**run_params)
            
            # Wait for completion
            run = self.wait_for_run(thread_id, run.id)
            
            # Get messages
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id
            )
            
            # Get the latest assistant message
            assistant_messages = [
                msg for msg in messages 
                if msg.role == "assistant"
            ]
            
            if not assistant_messages:
                return {
                    "answer": "No response received from the assistant.",
                    "is_valid": True,
                    "contexts": [],
                    "metadata": {}
                }
            
            latest_message = assistant_messages[0]
            
            # Extract the text value safely, handling potential format changes
            answer = ""
            if hasattr(latest_message, 'content') and latest_message.content:
                for content_item in latest_message.content:
                    if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                        answer = content_item.text.value
                        break
            
            return {
                "answer": answer,
                "is_valid": True,
                "contexts": [],
                "metadata": {
                    "thread_id": thread_id,
                    "run_id": run.id,
                    "retrieval_method": retrieval_method,
                    "k": k
                }
            }
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing your query: {str(e)}",
                "is_valid": False,
                "contexts": [],
                "metadata": {}
            }
    
    def wait_for_run(self, thread_id, run_id, timeout=300):
        """
        Wait for a run to complete.
        
        Args:
            thread_id: The thread ID
            run_id: The run ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            The completed run object
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                print(f"Run status: {run.status}")
                
                if run.status in ["completed", "failed", "cancelled", "expired"]:
                    return run
                    
                # Wait a bit before checking again
                time.sleep(1)
            except Exception as e:
                print(f"Error checking run status: {str(e)}")
                time.sleep(5)  # Wait longer on error
        
        # If we hit the timeout, try to cancel the run
        try:
            self.client.beta.threads.runs.cancel(
                thread_id=thread_id,
                run_id=run_id
            )
        except Exception:
            pass
            
        raise TimeoutError(f"Run {run_id} timed out after {timeout} seconds") 