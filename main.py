from langchain.tools.render import render_text_description
from langchain_community.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from typing import Union, List
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool

# from dotenv import load_dotenv
# from langchain.agents import tool
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.tools.render import render_text_description

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name:str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Could not find the tool with {tool_name}")


if __name__ == '__main__':
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
        Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        TOOLS:
        ------

        Assistant has access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        New input: {input}
        Thought: {agent_scratchpad}
    """
    print(render_text_description(tools))
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names = ", ".join(t.name for t in tools)
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x:x["input"], 
            "agent_scratchpad": lambda x:format_log_to_str(x["agent_scratchpad"])
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the text DOG?",
            "agent_scratchpad": intermediate_steps
        }
    )
    
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input


        
        observation = tool_to_use.func(tool_input)
        print(f"{observation}")
        intermediate_steps.append((agent_step, str(observation)))


    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the text DOG?",
            "agent_scratchpad": intermediate_steps
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)