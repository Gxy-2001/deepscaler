import argparse
import ast
import re
import os
import sys
from pebble import ProcessPool
from functools import partial
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel, Field
from timeout_decorator import timeout


TIMEOUT_SECONDS = 3
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

def truncate_string(text, max_length=4096, is_evalf=True):
    if is_evalf and isinstance(text, str):
        try:
            text_sympy = float(text)
            refine_text = str(round(text_sympy, 4)) + "\n"
            text = text if len(text) < len(refine_text) else refine_text
        except:
            pass

    if len(str(text)) > max_length:
        return str(text)[:max_length//2] + "..." + str(text)[-max_length//2:]
    return text

def extract_content(text):
    pattern = r'print\((.*?)\)'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        return ""
    return " ".join(matches)+":"


def __is_print_node(node: ast.AST) -> bool:
    
    if isinstance(node, ast.Expr) and \
        isinstance(node.value, ast.Call) and \
        isinstance(node.value.func, ast.Name) and \
        node.value.func.id == "print":
        return True
    elif isinstance(node, ast.If) or \
         isinstance(node, ast.While) or \
         isinstance(node, ast.For) or \
         isinstance(node, ast.FunctionDef):
        for sub_node in node.body:
            if __is_print_node(sub_node):
                return True
    return False


def find_print_node(body: List[ast.AST]) -> List[int]:
    """Find the python print node in the tree.body.

    Args:
        body (List[ast.AST]): The body of the AST

    Returns:
        List[int]: The index of the python print node
    """
    print_index = []
    for idx, node in enumerate(body):
        if __is_print_node(node):
            print_index.append(idx)
    return print_index


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
    

class PythonInterpreter(BaseModel):
    """A tool for running python code snippet."""

    name: str = "python_interpreter"
    description: str = (
        "A Python shell. Use this to execute python commands. "
    )
    description_zh: str = (
        "Python 交互式 shell。使用此工具来执行 Python 代码。"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    max_length: int = 4096
    is_evalf: bool = True
    args_schema: Type[BaseModel] = PythonInputs
    use_signals: bool = False 

    def _base_run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        def _sub_run(bodys):
            io_buffer = StringIO()
            module = ast.Module(bodys[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(bodys[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore

            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return True, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
                    else:
                        return True, truncate_string(ret, max_length=self.max_length, is_evalf=self.is_evalf)
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return False, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
            
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            print_indexs = find_print_node(tree.body)
            if len(print_indexs) == 0:
                print_indexs = [len(tree.body) - 1]
            ret_strs = []
            if len(print_indexs) == 1:
                run_flag, ret = _sub_run(tree.body)
                return f"{ret}"
            for start_idx, end_idx in zip([-1] + print_indexs, print_indexs):
                node_source = ast.get_source_segment(query, tree.body[end_idx])
                run_flag, ret = _sub_run(tree.body[start_idx + 1:end_idx + 1])
                ret_strs.append(f"{extract_content(node_source)} {ret}")
            return "".join(ret_strs)
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))
    
    def run(
        self,
        query: str,
    ) -> str:

        @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
        def base_run(query: str) -> str:
            return self._base_run(query)
        
        try:
            ret = base_run(query)
            return ret
        except Exception as e:
            print(e)
            print(" exec code error ")
            return "{}: {}".format(type(e).__name__, str(e))

def extract_program(result: str, last_only=False):
    program = ""
    start = False
    result = result.replace("<end_of_step>", "")
    #result = result.replace("<s>", "")
    for line in result.split("\n"):
        if line.find("<code>") != -1:
            if last_only:
                program = "" # only extract the last program
            else:
                program += "\n# ========\n"
            start = True
        elif line.find("<end_of_code>") != -1:
            start = False
        elif line.find("<end_of_step>") != -1:
            continue
        elif start:
            program += line + "\n"
    # maybe all output is a program
    if not program:
        program = result
    return program.strip()


def extract_code_segments(s: str) -> list:
    """
    从字符串中提取所有被 <code> 和 </code> 包含的子字符串

    参数:
        s (str): 输入的字符串

    返回:
        list: 被 <code> 和 </code> 包含的所有子字符串列表
    """
    # 使用非贪婪模式匹配 <code> 和 </code> 中间的内容
    return re.findall(r"<code>(.*?)</code>", s)

tool = PythonInterpreter(globals=globals(), locals=None)

def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool

def code_execution(
    code
) -> str:

    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(code) -> str:
        # Define tool
        tool_func = tool_wrapper(tool)
        # print(code)
        history_action_inputs = extract_code_segments(code)
        # print(history_action_inputs)
        observation = ""
        for history_ai in history_action_inputs:
            observation = tool_func(history_ai).strip()
        
        del tool_func
        return observation
    
    try:
        observation = _code_execution(code)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation

def batch_apply(codes):
    outputs = []
    with ProcessPool(max_workers=os.cpu_count() - 8) as pool:
    #with ProcessPool(max_workers=1) as pool:
        executor = partial(
            code_execution
        )
        future = pool.map(executor, codes, timeout=15)
        iterator = future.result()
        
        while True:
            try:
                result = next(iterator)
                outputs.append(result)
            except StopIteration:
                break
            except Exception as error:
                outputs.append("An error occurred, no output result.") 
                print("process error",error)
    outputs = [output + "</output>" for output in outputs]
    return outputs

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--testcase', type=str, default="```python\nprint(1)\nprint(2)\n```")
    # input args
    args = args.parse_args()
    return args

if __name__ == "__main__":
    # args = parse_args()
    # tool = PythonInterpreter(globals=globals(), locals=None)
    # print(tool.run(args.testcase))
    # sys.exit(0)

    # print(code_execution("123<code>print('code1')</code>3456<code>print('code2')</code>"))

    print(batch_apply([
        "123<code>print('code1')</code>3456<code>print('code2')</code>",
        "123<code>print('code1')</code>3456<code>print('code1')</code>"
    ]))