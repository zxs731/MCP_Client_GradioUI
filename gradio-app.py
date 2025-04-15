import asyncio  
from contextlib import AsyncExitStack  
import json  
import os  
import httpx  
import gradio as gr  
from dotenv import load_dotenv  
from mcp import ClientSession, StdioServerParameters  
from mcp.client.stdio import stdio_client  
from mcp.client.sse import sse_client  
from openai import AsyncAzureOpenAI  
import nest_asyncio
 
# 在Streamlit应用开始处应用nest_asyncio
nest_asyncio.apply()

# Load environment variables  
load_dotenv("./azureopenai.env")  
model = os.getenv("model")  
  
class MCPClient:  
    def __init__(self):  
        self.session = None  
        self.sessions = {}  
        self.exit_stack = AsyncExitStack()  
        self.tools = []  
        self.messages = []  
        self.client = AsyncAzureOpenAI(  
            azure_endpoint=os.environ["base_url"],  
            api_key=os.environ["api_key"],  
            api_version="2024-05-01-preview",  
            http_client=httpx.AsyncClient(verify=False)  
        )
        self.is_connected = False  
  
    async def cleanup(self):  
        await self.exit_stack.aclose()  
  
    async def connect_to_server(self): 
        if self.is_connected:
            return 
        with open("mcp_server_config.json", "r") as f:  
            config = json.load(f)  
        conf = config["mcpServers"]  
        self.tools = []  
        for key in conf.keys():  
            v = conf[key]  
            print(v)
            session = None  
            if "baseUrl" in v and v['isActive']:  
                server_url = v['baseUrl']  
                sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))  
                write, read = sse_transport  
                session = await self.exit_stack.enter_async_context(ClientSession(write, read))  
            elif "command" in v and v['isActive']:  
                command = v['command']  
                args = v['args']  
                server_params = StdioServerParameters(command=command, args=args, env=None)  
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))  
                stdio1, write1 = stdio_transport  
                session = await self.exit_stack.enter_async_context(ClientSession(stdio1, write1))  
  
            if session:  
                await session.initialize()  
                response = await session.list_tools()  
                tools = response.tools  
                for tool in tools:  
                    self.sessions[tool.name] = session  
                self.tools += tools  
        self.is_connected = True
        print("tools loaded!")
        print("MCPClient connected to server!")
  
    async def run_conversation(self, messages, tools, think_handle=None, content_handle=None):  
        response_message = await self.client.chat.completions.create(  
            model=model,  
            messages=messages,  
            tools=tools,  
            stream=True  
        )  
        content = ''  
        function_list = []  
        async for chunk in response_message:  
            if chunk and len(chunk.choices) > 0:  
                chunk_message = chunk.choices[0].delta  
                if chunk_message.content:  
                    content += chunk_message.content
                    if content_handle:
                        content_handle(chunk_message.content)  
                if chunk_message.tool_calls:  
                    for tool_call in chunk_message.tool_calls:  
                        if len(function_list) < tool_call.index + 1:  
                            function_list.append({'name': '', 'args': '', 'id': tool_call.id})  
                        if tool_call and tool_call.function.name:  
                            function_list[tool_call.index]['name'] += tool_call.function.name  
                        if tool_call and tool_call.function.arguments:  
                            function_list[tool_call.index]['args'] += tool_call.function.arguments  
                            
        if len(function_list) > 0:  
            findex = 0  
            tool_calls = []  
            temp_messages = []  
            for func in function_list:  
                function_name = func["name"]  
                print(function_name)
                function_args = func["args"]
                function_args = json.loads(function_args)
                toolid = func["id"]  
                if function_name != '': 
                    # 执行工具调用
                    print(f"⏳MCP: [Calling tool {function_name} with args {function_args}]")
                    function_response = await self.sessions[function_name].call_tool(function_name, function_args)
                    print(f"⏳MCP Done: [Calling tool {function_name} with args {function_args}]")
                    if think_handle:
                        think_handle(f"⏳MCP: [Calling tool {function_name} with args {function_args}]\r\nResult: {function_response.content}\r\n")

                    tool_calls.append({"id": toolid, "function": {"arguments": func["args"], "name": function_name}, "type": "function", "index": findex})  
                    temp_messages.append({  
                        "tool_call_id": toolid,  
                        "role": "tool",  
                        "name": function_name,  
                        "content": function_response.content,  
                    })  
                    findex += 1  
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})  
            for m in temp_messages:  
                messages.append(m)  
            return await self.run_conversation(messages, tools, think_handle, content_handle)  
        elif content != '':  
            messages.append({"role": "assistant", "content": content})  
            return messages[-1]  
  
    async def process_query(self, query: str,feedback=None): 
        await self.connect_to_server()
        print("aaa")
        self.messages.append({"role": "user", "content": query})  
        messages = self.messages[-20:]  
        available_tools = [{  
            "type": "function",  
            "function": {  
                "name": tool.name,  
                "description": tool.description,  
                "parameters": tool.inputSchema  
            }  
        } for tool in self.tools]  
        print("bbb")
        reply_message = await self.run_conversation(messages, available_tools,content_handle=feedback)  
        print(reply_message)
        self.messages.append(reply_message)  
        return reply_message  
  


client = MCPClient() 

  
# Gradio UI
with gr.Blocks() as demo:  
    # 添加标题  
    gr.Markdown("<h1 style='text-align: center;'>MCP Agent</h1>")  
    chatbot = gr.Chatbot(type="messages",avatar_images=['./2.png','./1.png'])  
    msg = gr.Textbox(label="You:",placeholder="Type your message here...")  
    clear = gr.ClearButton([msg, chatbot])  
    
    async def respond(message, chat_history): 
        def feedback(content):
            if chat_history[-1]["content"] == "waiting...":
                chat_history[-1]["content"] = ""
            chat_history[-1]["content"] += content
            return "", chat_history
        if len(chat_history)>0 and chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] == "waiting...":
            response=await client.process_query(chat_history[-2]["content"],feedback=feedback)
            #bot_message = response["content"]
            #chat_history[-1]["content"]=bot_message
  
        return "", chat_history  
    
    def fill_user_message(message, chat_history):
        if len(chat_history)>0 and  chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] == "waiting...":
            return
        chat_history.append({"role": "user", "content": message})  
        chat_history.append({"role": "assistant", "content": "waiting..."})  
        return "", chat_history
    
    chatbot.change(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(fill_user_message, [msg, chatbot], [msg, chatbot]) 
    
  
if __name__ == "__main__":  
    demo.launch()  