from venv import create

from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

# Functions make a chatbot into an agent
@tool
def add_task(task, desc=None):
    """
    Add a new task to the user's task list.
    Use this when the user wants to add or create a task
    :return:
    """
    todoist.add_task(content=task,
                     description=desc)

#initialize the llm
tools = [add_task]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3 #closer to 0 is less creative
)

#initialize prompt for the llm
system_prompt = "You are a helpful assistant. You will help the user add tasks."

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"), #should be specifically after system prompt
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"), #will hold intermediate agent messages
])

# chain used to provide prompt to llm for chat botting
# chain = prompt | llm | StrOutputParser()

#agent takes in the llm, tools/functions, and prompt to perform agent tasks
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# response = chain.invoke({"input":user_input})

# print(response)

history = []
while True:
    user_input = input("User: ").strip()
    if not user_input:
        continue

    try:
        response = agent_executor.invoke({"input": user_input,
                                          "history":history})
        print(response)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response["output"]))
    except Exception as e:
        print("Error:", e)

