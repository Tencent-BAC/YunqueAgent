"""
File Parser Module
Supports parsing various file types: PDF, DOCX, PPTX, TXT, CSV, XLSX, ZIP, etc.
"""
import sys
import os
import re
import time
import base64
import urllib
import requests
import zipfile
from typing import Optional


class FileParser:
    """
    File parser class for handling various file types.
    Supports: PDF, DOCX, PPTX, TXT, CSV, XLSX, ZIP, and more.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def download_url_to_local(url_type_info: dict, save_dir: str = "./workspace/files", max_retries: int = 2) -> dict:
        """
        Download file from URL to local directory.
        
        Filename extraction priority:
        1. file_name from url_type_info (already parsed by get_url_type_by_get_request)
        2. Generate fallback name from URL hash + file_type
        
        Args:
            url_type_info: Dictionary containing URL type information from get_url_type_by_get_request
            save_dir: Directory to save the downloaded file
            max_retries: Maximum number of retry attempts for downloading the file
            
        Returns:
            dict: {
                'status': 'success' | 'error',
                'file_path': str (absolute path to downloaded file),
                'file_name': str (filename),
                'error_message': str (only when status='error')
            }
        """
        # Extract the final URL (after any redirects) from url_type_info
        download_url = url_type_info.get('redirect_url')
        if not download_url:
            return {
                'status': 'error',
                'error_message': "No redirect_url found in url_type_info"
            }
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Priority 1: Use file_name from url_type_info (already parsed and cleaned)
        filename = url_type_info.get('file_name', '')
        
        # Priority 2: Generate fallback name using hash + file_type
        if not filename:
            file_type = url_type_info.get('file_type', 'bin')
            filename = f"downloaded_{abs(hash(download_url))}.{file_type}"
        
        # Sanitize filename (remove invalid characters for filesystem)
        filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        filepath = os.path.join(save_dir, filename)
        
        # Retry logic for download
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"[download] Retry attempt {attempt}/{max_retries} for {download_url}")
                    time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s...
                
                # Download file from the final redirect URL
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return {
                    'status': 'success',
                    'file_path': filepath,
                    'file_name': filename
                }
            except Exception as e:
                last_error = str(e)
                print(f"[download] Attempt {attempt + 1} failed: {last_error}")
                if attempt >= max_retries:
                    break
        
        # All retries failed
        return {
            'status': 'error',
            'error_message': f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
        }
    
    @staticmethod
    def get_url_type_by_get_request(url: str) -> dict:
        """
        Determine URL type: downloadable file or web page.
        
        Detection priority:
        1. Content-Disposition header (most reliable)
        2. X-Raw-Download header (GitHub raw files, etc.)
        3. Content-Type header (MIME type mapping)
        4. URL file extension (fallback)
        
        Args:
            url: The URL to check
            
        Returns:
            dict: {
                'url_type': 'file' | 'html' | 'unknown',
                'file_type': str (only when url_type='file'),
                'file_name': str (extracted filename when available),
                'content_type': str,
                'content_disposition': str,
                'redirect_url': str
            }
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
            content_type = response.headers.get('Content-Type', '').lower()
            content_disposition = response.headers.get('Content-Disposition', '')
            raw_download_url = response.headers.get('x-raw-download')
            
            # Determine redirect URL: use response.url (final URL after redirects) if different from original
            final_url = response.url if response.url != url else url
            
            # Base response structure
            res = {
                'content_type': content_type,
                'content_disposition': content_disposition,
                'redirect_url': final_url
            }
            
            # Method 1: Check Content-Disposition header (most reliable for file downloads)
            if content_disposition and 'attachment' in content_disposition.lower():
                res['url_type'] = 'file'
                
                # Extract filename from Content-Disposition
                filename_match = re.search(
                    r'filename\*?=(?:UTF-8\'\')?"\'?([^"\'\;\r\n]+)["\'\?', 
                    content_disposition, 
                    re.IGNORECASE
                )
                if filename_match:
                    filename = urllib.parse.unquote(filename_match.group(1))
                    res['file_name'] = filename
                    if '.' in filename:
                        res['file_type'] = filename.split('.')[-1].lower()
                
                response.close()
                return res
            
            # Method 2: Check X-Raw-Download header (GitHub raw files, etc.)
            if raw_download_url:
                res['url_type'] = 'file'
                # Use the raw download URL as the redirect URL
                res['redirect_url'] = raw_download_url
                
                # Extract filename and extension from raw download URL
                from file_tools.utils import get_basename_from_url
                filename = get_basename_from_url(raw_download_url)
                if filename:
                    res['file_name'] = filename
                
                ext = FileParser._extract_extension_from_url(raw_download_url)
                if ext:
                    res['file_type'] = ext
                
                response.close()
                return res
            
            # Method 3: Check Content-Type header (MIME type mapping)
            file_type_mapping = {
                'application/zip': 'zip',
                'application/x-zip-compressed': 'zip',
                'application/pdf': 'pdf',
                'application/vnd.ms-excel': 'xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'text/csv': 'csv',
                'application/json': 'json',
                'application/xml': 'xml',
                'text/xml': 'xml',
                'application/octet-stream': None,  # Generic binary, needs further detection
            }
            
            for mime_type, ext in file_type_mapping.items():
                if mime_type in content_type:
                    res['url_type'] = 'file'
                    
                    # Try to extract filename from URL
                    from file_tools.utils import get_basename_from_url
                    filename = get_basename_from_url(url)
                    if filename:
                        res['file_name'] = filename
                    
                    if ext:
                        res['file_type'] = ext
                    else:
                        # For octet-stream, infer from URL extension
                        url_ext = FileParser._extract_extension_from_url(url)
                        if url_ext:
                            res['file_type'] = url_ext
                    
                    response.close()
                    return res
            
            # Method 4: Infer from URL file extension
            url_ext = FileParser._extract_extension_from_url(url)
            if url_ext and url_ext in ['zip', 'csv', 'pdf', 'xls', 'xlsx', 'json', 'txt', 'xml']:
                res['url_type'] = 'file'
                res['file_type'] = url_ext
                
                # Extract filename from URL
                from file_tools.utils import get_basename_from_url
                filename = get_basename_from_url(url)
                if filename:
                    res['file_name'] = filename
                
                response.close()
                return res
            
            # Method 5: Check for HTML or plain text content type
            if 'text/html' in content_type or 'text/plain' in content_type:
                res['url_type'] = 'html'
                response.close()
                return res
            
            # Default to HTML
            res['url_type'] = 'html'
            response.close()
            return res
        
        except requests.RequestException as e:
            return {'url_type': 'unknown', 'error': str(e), 'redirect_url': url}
    
    @staticmethod
    def _extract_extension_from_url(url: str) -> Optional[str]:
        """从 URL 中提取文件扩展名"""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        if '.' in path:
            ext = path.split('.')[-1].lower()
            ext = ext.split('?')[0].split('#')[0]
            if ext and len(ext) <= 10:
                return ext
        return None
    
    @staticmethod
    def parse_zip(file_path: str):
        """解析ZIP文件并解压"""
        extracted_files = []
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_dir = os.path.splitext(file_path)[0]
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
                extracted_files = [os.path.join(extract_dir, name) for name in zip_ref.namelist()]
            return {
                'status': 'success',
                'extracted_files': extracted_files
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    @staticmethod
    def robust_load_csv(file_path: str):
        """加载CSV文件，自动检测元数据和表头位置"""
        import pandas as pd
        import csv
        
        metadata_lines = []
        header_row_index = 0
        
        # 设定预读行数，如果文件不够长，read_rows 会包含所有行
        scan_lines = 50 
        read_rows = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                # 循环读取，直到达到 scan_lines 或者文件结束
                for _ in range(scan_lines):
                    try:
                        row = next(reader)
                        read_rows.append(row)
                    except StopIteration:
                        # 文件读完了（行数 < 50），直接跳出
                        break
                
            # 如果文件是空的
            if not read_rows:
                return "", pd.DataFrame(), 0

            # 分析列数
            col_counts = [len(r) for r in read_rows]
            print(f"Pre-scan column counts per row: {col_counts}")
            max_cols = max(col_counts)
            
            # 策略：找到第一个达到最大列数的行作为表头
            # 比如：第1行2列，第2行2列，第3行70列 -> 索引2是表头
            header_row_index = col_counts.index(max_cols)
            print(f"Determined header row index: {header_row_index}")
            
            # 如果最大列数和第一行列数一样，说明没有元数据，或者第一行就是表头
            if header_row_index == 0:
                metadata_str = ""
            else:
                # 提取元数据
                for i in range(header_row_index):
                    # 去除空字符串
                    row_content = [x.strip() for x in read_rows[i] if x and x.strip()]
                    if row_content:
                        metadata_lines.append(" | ".join(row_content))
                
                metadata_str = "### File Metadata\n" + "\n".join([f"- {line}" for line in metadata_lines]) + "\n\n"

        except Exception as e:
            print(f"Warning: Pre-scan failed: {e}, falling back to default.")
            return "", pd.read_csv(file_path), 0

        # 使用 Pandas 读取
        try:
            print(f"Loading CSV with header at row index: {header_row_index}")
            # header=header_row_index 告诉 pandas 跳过前面的元数据行
            df = pd.read_csv(file_path, header=header_row_index, skip_blank_lines=False)
            print(f"CSV loaded with shape: {df.shape}")
            print(f"CSV columns: {df.columns.tolist()}")
        except Exception:
            # 兜底策略
            df = pd.read_csv(file_path)
            
        return metadata_str, df, header_row_index
    
    @staticmethod
    def parse_small_csv(data: tuple):
        """解析小型CSV文件（显示全部内容）"""
        try:
            metadata, df, header_row_index = data
            
            csv_summary = []
            if metadata:
                csv_summary.append(metadata)
                
            # --- Basic Information ---
            csv_summary.append(f"### Basic Information")
            csv_summary.append(f"- Number of rows: {df.shape[0]}")
            csv_summary.append(f"- Number of columns: {df.shape[1]}")
            csv_summary.append(f"- Column names: {', '.join(map(str, df.columns.tolist()))}")
            csv_summary.append(f"- Header row index (0-based, absolute index in raw file including blank lines): {header_row_index}")
            csv_summary.append("\n")
            
            # --- File Content (Full) ---
            csv_summary.append(f"### File Content (Full)")
            csv_summary.append(df.to_markdown(index=False))
            return "\n".join(csv_summary)
        
        except Exception as e:
            print(f"Error parsing small CSV: {str(e)}")
            return f"Error parsing CSV: {str(e)}"
    
    @staticmethod
    def parse_large_csv(data: tuple, max_lines: int = 10):
        """解析大型CSV文件（只显示前N行）"""
        try:
            metadata, df, header_row_index = data
            
            csv_summary = []
            if metadata:
                csv_summary.append(metadata)
                
            csv_summary.append(f"### Basic Information")
            csv_summary.append(f"- Number of rows: {df.shape[0]}")
            csv_summary.append(f"- Number of columns: {df.shape[1]}")
            csv_summary.append(f"- Column names: {', '.join(map(str, df.columns.tolist()))}")
            csv_summary.append(f"- Header row index (0-based, absolute index in raw file including blank lines): {header_row_index}")
            csv_summary.append("\n")
            
            csv_summary.append(f"### First {max_lines} rows")
            csv_summary.append(df.head(max_lines).to_markdown(index=False))
            csv_summary.append("\n")
            
            return "\n".join(csv_summary)
        
        except Exception as e:
            print(f"Error parsing large CSV: {str(e)}")
            return f"Error parsing large CSV: {str(e)}"
    
    @staticmethod
    def parse_csv(file_path: str, threshold_rows: int = 300):
        """解析CSV文件，根据大小选择不同的解析策略"""
        try:
            metadata, df, header_row_index = FileParser.robust_load_csv(file_path)
            if df.shape[0] <= threshold_rows:
                return FileParser.parse_small_csv(data=(metadata, df, header_row_index))
            else:
                return FileParser.parse_large_csv(data=(metadata, df, header_row_index))

        except Exception as e:
            return f"Error parsing CSV: {str(e)}"
    
    @staticmethod
    def parse_txt(file_path: str, max_chars: int = 5000) -> str:
        """解析TXT文本文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if len(content) > max_chars:
                return f"### TXT File Content (Snippet - first {max_chars} characters out of {len(content)} total)\n\n{content[:max_chars]}\n\n... (content truncated)"
            else:
                return f"### TXT File Content (Full - {len(content)} characters)\n\n{content}"
        except Exception as e:
            print(f"Error parsing TXT: {str(e)}")
            return f"Error parsing TXT: {str(e)}"
    
    @staticmethod
    def file_to_base64(file_path: str) -> str:
        """将文件内容转换为Base64编码"""
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                
            # Base64 encode the file content
            base64_bytes = base64.b64encode(file_content)
            
            # Convert bytes to string
            base64_string = base64_bytes.decode('utf-8')
            
            return base64_string
        
        except Exception as e:
            print(f"[file_to_base64] Base64 encoding failed: {e}")
            return ""
