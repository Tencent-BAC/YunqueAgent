import os

from openai import OpenAI
from sandbox_fusion import run_code, RunCodeRequest
from prompt import DEEPANALYZER_PROMPT

# Import FileParser class from tool_file
from tool_file import FileParser


class DeepAnalyzer:
    
    def __init__(self):
        pass
    
    def download(self, url_type_info: dict):
        # Download the file to local using the redirect_url from url_type_info
        download_info = FileParser.download_url_to_local(url_type_info, max_retries=2)
        if download_info['status'] != 'success':
            print(f"[deep_analysis] File download failed: {download_info.get('error_message', '')}")
            return download_info
        print(f"[deep_analysis] File downloaded successfully: {download_info['file_path']}")
        return download_info
    
    def call(self, download_info: dict, goal: str, question: str):
        file_path = download_info['file_path']
        file_name = download_info['file_name']
        file_ext = os.path.splitext(file_name)[1].lower()
        
        print(f"[deep_analysis] DEBUG: Recieved file for analysis - Path: {file_path}, Name: {file_name}, Extension: {file_ext}, Goal: {goal}, Question: {question}")
        
        if file_ext == '.zip':
            return self._analyze_zip_file(
                file_path,
                file_name,
                goal,
                question,
                max_retry=3
            )
        
        elif file_ext == '.csv':
            return self._analyze_csv_file(
                file_path,
                file_name,
                goal,
                question,
                max_retry=3
            )
            
        elif file_ext == '.txt':
            return self._analyze_txt_file(
                file_path,
                file_name,
                goal,
                question,
                max_retry=3
            )
            
        else:
            return f"[deep_analysis] Unsupported file type: {file_ext}. Currently only .zip and .csv files are supported."       
    
    
    def call_server(self, msgs, max_retries=3):
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("DEEPANALYZER_MODEL_NAME", "gemini-3-pro")
        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
        )
        
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=1.0
                )
                content = chat_response.choices[0].message.content
                return content
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
        return "Error: All retries to call the server failed."
    
    
    def call_python(self, code: str, files: dict):
        code_result = run_code(
                    RunCodeRequest(
                        code=code,
                        language='python',
                        run_timeout=30,
                        files=files
                    ),
                    max_attempts=1,
                    client_timeout=30,
                    endpoint='http://127.0.0.1:8080'
                )
        
        return code_result
    
    
    def _analyze_zip_file(self, file_path: str, file_name: str, goal: str, question: str, max_retry: int):
        # If the file is a zip, extract it first
        parse_info = FileParser.parse_zip(file_path)
        if parse_info['status'] != 'success':
            return f"[deep_analysis] Zip file extraction failed: {parse_info.get('error_message', '')}"
        
        extracted_files = parse_info['extracted_files']
        if not extracted_files:
            return "[deep_analysis] Zip file is empty."
        
        results = []
        skipped_files = []
        
        for extracted_file in extracted_files:
            print(f"[deep_analysis] Processing extracted file: {extracted_file}")
            extracted_file_name = os.path.basename(extracted_file)
            extracted_file_ext = os.path.splitext(extracted_file_name)[1].lower()
            
            if extracted_file_ext == '.csv':
                try:
                    result = self._analyze_csv_file(
                        extracted_file,
                        extracted_file_name,
                        goal,
                        question,
                        max_retry
                    )
                    results.append(f"### Deep analysis for extracted file {extracted_file} in zip file {file_name}:\n{result}")
                except Exception as e:
                    results.append(f"### Deep analysis for extracted file {extracted_file} in zip file {file_name}:\nError: {e}")
            else:
                skipped_files.append(f"{extracted_file_name} ({extracted_file_ext})")
                
        final_output = []
        
        if results:
            final_output.append("## Analyzed Files")
            final_output.append("\n\n".join(results))
            
        if skipped_files:
            final_output.append("## Skipped Files")
            final_output.append("The following files were skipped due to unsupported file types:\n" + ", ".join(skipped_files))
            
        return "\n\n".join(final_output)
    
    
    def _analyze_csv_file(self, file_path: str, file_name: str, goal: str, question: str, max_retry: int):
        
        try:
            file_content = FileParser.parse_csv(file_path)
            print(f"[deep_analysis] CSV parsed. Content preview: \n{file_content[:1000]}...")
        except Exception as e:
            return f"CSV parsing failed: {str(e)}"
        
        message = [{
            "role":"user",
            "content": DEEPANALYZER_PROMPT.format(
                file_name=file_name,
                file_content=file_content,
                goal=goal,
                question=question
            )
        }]
        
        # First LLM call
        try:
            response = self.call_server(message)
            print(f"Deep Analysis Response:\n{response}")
        except Exception as e:
            return f"LLM call failed: {str(e)}"
        
        if "<code>" not in response or "</code>" not in response:
            return response
        
        # Execute code with retries
        return self._execute_code_with_retries(
            message=message,
            initial_response=response,
            file_path=file_path,
            file_name=file_name,
            max_retry=max_retry
        )
    
    
    def _analyze_txt_file(self, file_path: str, file_name: str, goal: str, question: str, max_retry: int):
        
        try:
            file_content = FileParser.parse_txt(file_path)
            print(f"[deep_analysis] TXT parsed. Content preview: \n{file_content[:1000]}...")
        except Exception as e:
            return f"TXT parsing failed: {str(e)}"
        
        message = [{
            "role":"user",
            "content": DEEPANALYZER_PROMPT.format(
                file_name=file_name,
                file_content=file_content,
                goal=goal,
                question=question
            )
        }]
        
        # First LLM call
        try:
            response = self.call_server(message)
            print(f"Deep Analysis Response:\n{response}")
        except Exception as e:
            return f"LLM call failed: {str(e)}"
        
        if "<code>" not in response or "</code>" not in response:
            return response
        
        # Execute code with retries
        return self._execute_code_with_retries(
            message=message,
            initial_response=response,
            file_path=file_path,
            file_name=file_name,
            max_retry=max_retry
        )
        
        
    def _execute_code_with_retries(self, message: list, initial_response: str, file_path: str, file_name: str, max_retry: int):
        # prepare file for code execution
        file_base64 = FileParser.file_to_base64(file_path)
        files = {file_name: file_base64}
        
        current_response = initial_response
        retry_count = 0
        
        while retry_count <= max_retry:
            print(f"\n[deep_analysis] {'Initial execution' if retry_count == 0 else f'Retry {retry_count}/{max_retry}'}")
            
            code_snippet = self._extract_code(current_response)
            if not code_snippet:
                return current_response
            
            try:
                code_result = self.call_python(code_snippet, files)
                print(f"[deep_analysis] Code execution result:\n{code_result}")
            except Exception as e:
                code_result = f"Exception during code execution: {str(e)}"
                print(f"[deep_analysis] {code_result}")
            
            is_success = "status=<RunStatus.Success" in str(code_result)
            
            message.append({"role":"assistant","content": current_response})
            
            if is_success:
                message.append({"role":"user","content": f"The code executed successfully. Based on the result below, please provide a clear and concise summary to answer the question.\n\nCode execution result:\n{code_result}"})
                
                try:
                    final_response = self.call_server(message)
                    return final_response
                except Exception as e:
                    return f"Code executed successfully, but failed to generate summary: {str(e)}\n\nRaw result:\n{code_result}"
                
            else:
                if retry_count >= max_retry:
                    return f"Code execution failed after {max_retry} retries. Last error:\n{code_result}"
                
                message.append({"role":"user","content": f"The code execution failed with the following error. Please analyze the error and provide a corrected version of the code wrapped in <code></code> tags.\n\nCode execution result:\n{code_result}"})
                
                try:
                    current_response = self.call_server(message)
                except Exception as e:
                    return f"Failed to get corrected code from LLM: {str(e)}"
                
                retry_count += 1
                
        return f"[deep_analysis] Unexpected error in retry loop."
    
    
    def _extract_code(self, response: str) -> str:
        if "<code>" not in response or "</code>" not in response:
            return ""
        try:
            code_start = response.index("<code>") + len("<code>")
            code_end = response.index("</code>")
            code_snippet = response[code_start:code_end].strip()
            return code_snippet
        except Exception as e:
            print(f"[deep_analysis] Error extracting code: {str(e)}")
            return ""
        
    