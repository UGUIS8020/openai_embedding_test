import json
import nbformat
from pathlib import Path
import os

def extract_text_from_notebook(notebook_path):
    """
    ipynbファイルからテキストを抽出する関数
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    extracted_texts = []
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            extracted_texts.append(('markdown', cell.source))
        elif cell.cell_type == 'code':
            # コードセルの場合、コードと出力の両方を抽出
            code = cell.source
            outputs = []
            for output in cell.outputs:
                if 'text' in output:
                    outputs.append(output.text)
                elif 'data' in output and 'text/plain' in output.data:
                    outputs.append(output.data['text/plain'])
            
            extracted_texts.append(('code', code))
            if outputs:
                extracted_texts.append(('output', '\n'.join(outputs)))
    
    return extracted_texts

def save_extracted_text(extracted_texts, output_path):
    """
    抽出されたテキストをファイルに保存する関数
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell_type, content in extracted_texts:
            f.write(f"[{cell_type}]\n")
            f.write(content)
            f.write("\n\n")

def process_notebook(notebook_path, output_folder):
    """
    ノートブックを処理し、テキストを抽出して保存する関数
    """
    notebook_name = Path(notebook_path).stem
    output_path = Path(output_folder) / f"{notebook_name}_extracted.txt"
    
    extracted_texts = extract_text_from_notebook(notebook_path)
    save_extracted_text(extracted_texts, output_path)
    print(f"Extracted text from {notebook_path} saved to {output_path}")

def process_lang_folder(lang_folder, output_folder):
    """
    langフォルダー内の全てのipynbファイルを処理する関数
    """
    # 出力フォルダが存在しない場合は作成
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # langフォルダー内のすべてのipynbファイルを取得
    ipynb_files = list(Path(lang_folder).glob('*.ipynb'))
    
    if not ipynb_files:
        print(f"No ipynb files found in {lang_folder}")
        return
    
    for ipynb_file in ipynb_files:
        process_notebook(ipynb_file, output_folder)

# メイン処理
if __name__ == "__main__":
    current_dir = Path.cwd()
    lang_folder = current_dir / "lang"
    output_folder = current_dir / "ipynb_conversion"
    
    process_lang_folder(lang_folder, output_folder)
    print(f"All notebooks processed. Results saved in {output_folder}")