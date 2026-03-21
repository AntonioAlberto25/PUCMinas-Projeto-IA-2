import subprocess
import tempfile
import os

def execute_tests(converted_code: str, test_code: str, target_lang: str) -> dict:
    """
    Grava o código convertido e o código de teste num diretório temporário,
    executa os testes e retorna o resultado formatado.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Definir nomes de arquivos e comandos com base na linguagem alvo
        if target_lang.lower() in ["javascript", "js", "node"]:
            source_file = os.path.join(temp_dir, "converted_target.js")
            test_file = os.path.join(temp_dir, "test.js")
            command = ["node", test_file]
        else:
            # Padrão: Python
            source_file = os.path.join(temp_dir, "converted_target.py")
            test_file = os.path.join(temp_dir, "test_target.py")
            command = ["pytest", test_file, "-v"]
            
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(converted_code)
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
            
        try:
            result = subprocess.run(
                command,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            return {
                "success": success,
                "output": output
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Test execution timed out."
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Test execution failed: {str(e)}"
            }
