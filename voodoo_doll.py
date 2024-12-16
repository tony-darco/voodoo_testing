from pathlib import Path
import json
import os
import sys
import subprocess
import requests
from dotenv import load_dotenv
import base64
from datetime import datetime
import time
from operator import itemgetter
from typing import Literal


#lang graph implementation packages
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

#lang chain implementation packages
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate


#few shot packages
from langchain_core.prompts import FewShotChatMessagePromptTemplate

#embedding packages
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb


#mongodb package
import pymongo

import logging

load_dotenv()

class Magic:
    def __init__(self, max_iterations:int = 8, framework:str = "Pytest", show_output:bool = True, show_iterations:bool = True, show_eval:bool = False):

        load_dotenv()
        self._set_fileStructure()
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(
            filename=f'{self.currentdir}/logfile.log',
            level=logging.INFO,
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.show_output:bool = show_output
        self.show_iterations:bool = show_iterations
        self.show_eval:bool = show_eval

        self.ERROR_MESS_POS = 1
        self.ERRORED_POS = 0

        self.app_key = os.environ.get('APP_KEY')

        self.set_database(mongo_url = os.environ.get('MONGO_URL'),mongo_collection_name=os.environ.get('MONGO_COLLECTION_NAME'), mongo_db_name=os.environ.get('MONGO_DB_NAME'))

        self.max_iterations:int = max_iterations

        self._get_llm()

        self.framework = framework


    def set_framework(self, framework:str):
        self.framework = framework

    def set_database(self, mongo_url:str = 'mongodb://localhost:27017/', mongo_collection_name:str = 'Records', mongo_db_name:str = 'MAGIC_record') -> None:
        self.db_url = mongo_url
        self.collection_name = mongo_collection_name
        self.db_name = mongo_db_name
        


    def _set_fileStructure(self):
        self.file_name_prefix = "gen_code_"

        self.currentdir = Path(__file__).parent.resolve()

        self.code_file_name_path = f'{self.currentdir}/generated_code/{self.file_name_prefix}'

        self.example_code_dir = f'{self.currentdir}/documents/examples/'
        self.chunks_dir = f'{self.currentdir}/documents/chunks/'

        self.spec_sheet = f'{self.currentdir}/documents/meraki-api-spec.json'

        self.chroma_db_dir = f'{self.currentdir}/databases/chroma_vector_db/'
        self.mongodb_dir = f'{self.currentdir}/databases/mongo_record/'
        self.chroma_db_dir = f'{self.currentdir}/databases/chroma_vector_db'

    def _get_token(self):
        """
        Get the token from the Cisco Identity Service Engine
        """
        
        client_id = os.environ.get('CLIENT_ID')
        client_secret = os.environ.get('CLIENT_SECRET')
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }
        token_response = requests.request("POST", url, headers=headers, data=payload)
        return token_response

    def _get_llm(self):

        token_response = self._get_token()

        openai_api_key = token_response.json()["access_token"]

        app_key = os.environ.get('APP_KEY')

        self.cisco_llm = AzureChatOpenAI(temperature=0,
        model ='gpt-4o-mini',
        openai_api_key = openai_api_key,
        api_version = "2023-12-01-preview",
        azure_endpoint = openai_api_base,
        user = json.dumps({ 'appkey': app_key })
        )

    def _get_examples(self):
        fixture_example = ''
        example_code = []
        
        files = os.listdir(self.example_code_dir)
        dir_list = [entry for entry in files if entry.endswith('.json') and os.path.isfile(os.path.join(self.example_code_dir, entry))]
        
        if self.route != 'negative_code_generation':
            dir_list = [example_file for example_file in dir_list if "neg" not in str(example_file)]
            

        for doc in dir_list:
            try:
                with open(f"{self.example_code_dir}{doc}", "r") as f:
                    fixture_example = f.read()
                    fixture_example = json.loads(fixture_example)
            except Exception as e:
                self.logger.exception("Failed to grab one_shot code: ", e)

            fixture_prompt = fixture_example["prompt"]
            fixture_output = fixture_example["output"]

            example_code += [
                {"input" : fixture_prompt, "output" : fixture_output},
            ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),

            ]
        )

        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=example_code,
        )
        
        

    def _create_prompt(self):
        self.context_summary_prompt = PromptTemplate.from_template(
            """Can you provide a comprehensive summary of the given API Document? The summary should include the endpoint and the method (GET, PUT, POST, DELETE) which is the main focus of the document. The Summary should list all the parameters by name, where they should be and the main ideas of the description, while also condensing the information into a concise format. Please ensure that the summary includes relevant details each property including the different options and data types and examples, while avoiding any unnecessary information or example certificates. Provide and example request body base on the context. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.

            here is the API document need to summarize: {context}
            """
            )

        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                        """You are a senior software engineer and have to develop tests for Meraki Dashboard API using the API documents. You have to account for the multiple options for each attribute that is being used in the put/post requests. Look closely at the API documentation and use the right attributes for the respective end-points paying extra attention to the required and optional attributes. The help for the API we are going to generate test cases is here: \n {context}. \n ------- \n {test_framework} testing the end-point according to the documentation provided. Assume that we have the keys, Organization ID, a list on containing a device serial in a .env file and leverage dotenv module to extract the keys and use the environment variables in the code. The variables are names: {list_env_var} Your Task: The output script should first implement a fixture to create the network, named MAGIC-INTERN_PROJ in a user specified organization using Meraki Dashboard API calls. Then follow these user instruction. {user_query}. The test method is going to then call the end-point : {endpoint} and check if the response is a 200. Teardown Delete the network after tests""",
                ),
                self.few_shot_prompt,
                ("placeholder", "{messages}"),
            ]
        )

        self.neg_code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                        """You are a senior Software engineer and are tasked with developing a series of tests for the Meraki Dashboard API, focusing specifically on generating negative test cases. Your objective is to ensure comprehensive coverage of each API endpoint by considering multiple attribute options and adhering closely to the API documentation. Context and Instructions: API Documentation: {context}. Carefully examine the provided API documentation to identify the required and optional attributes for each endpoint. Pay special attention to using the correct attributes in PUT/POST requests. Your primary goal is to create negative test cases: for the each of the following error responses: 400 - Bad Request: Ensure the request fails due to missing required parameters. 401 - Unauthorized: Simulate scenarios with incorrect API keys. 404 - Not Found: Attempt to access non-existent resources. For each error code create a test. Setup Instructions: Implement a fixture to create a network named "MAGIC-INTERN_PROJ" within a specified organization using the Meraki Dashboard API. Use the dotenv module to extract environment variables, including the API keys, Organization ID, and a device serial. These should be stored in a .env file with variables named as follows: {list_env_var}. Testing Framework: Use the {test_framework} to execute the tests according to the documentation provided. Endpoint Testing: The test method should call the specified endpoint: {endpoint}. Create a test that fulfill this request: {user_query}. Verify that the response is a error responses. If the attribute is optional, test the attribute using an invalid value rather than omitting it. After completing the tests, ensure the network is deleted to maintain a clean testing environment. .
""",
                ),
                self.few_shot_prompt,
                ("placeholder", "{messages}"),
            ]
        )

        self.code_evaluation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a senior Software engineer and been given the Code was generated by an LLM.Encountered an error message while running pytest code, and need help understanding what went wrong. Here is the error message:{error}. Can you provide a comprehensive summary of the error message and what might have caused this issue? If the error is caused by a not required attributes, recommend its omission. Different methods(POST,GET,PUT, Delete) use different endpoints. here are the required attributes {context}
                    """
                )
            ]
        )
        
        self.code_fix_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a senior Software engineer and been given the Code was generated by an LLM. The code has some errors that need fixing. Here are some insights about the error and how to fix them {evaluation}. Given the following error message make changes to the code to resolve the error.
                    """
                ), self.few_shot_prompt,
                ("placeholder", "{messages}"),
            ]
        )

        self.neg_code_fix_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a senior Software engineer and been given the Code was generated by an LLM. The code has some errors that need fixing. Your primary goal is to create negative test cases for the each of the following error responses: 400 - Bad Request: Ensure the request fails due to missing required parameters. 401 - Unauthorized: Simulate scenarios with incorrect API keys. 404 - Not Found: Attempt to access non-existent resources.  Verify that the response is a error responses. Here are some insights about the error and how to fix them {evaluation}. Given the following error message make changes to the code to resolve the error. Make sure to test all error codes.
                    """
                ), self.few_shot_prompt,
                ("placeholder", "{messages}"),
            ]
        )


    #defined output structure
    class code_struc(BaseModel):
        """Schema for code output"""
        prefix: str = Field(description="Description of the problem and approach")
        code: str = Field(description="Import Statements and Code blocks")

    class summary_struc(BaseModel):
        """Schema for summary output"""
        prefix: str = Field(description="Description of the problem and approach")
        summary: str = Field(description="Summary of API document")


    def _parse_output(self, solution):
        """When we add 'include_raw=True' to structured output,
        it will return a dict w 'raw', 'parsed', 'parsing_error'."""

        return solution["parsed"]


    class _GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            error : Binary flag for control flow to indicate whether test error was tripped
            messages : With endpoint, error messages, reasoning
            generation : Code solution
            iterations : Number of tries
        """

        error: List
        messages: List
        context: str
        generation: str
        iterations: int
        evaluate: str
        
    def _intent(self) -> None:
        
        route_system = "Route the user's query to either the code_generation or negative_code_generation."
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", route_system),
                ("human", "{user_query}"),
            ]
        )

        class RouteQuery(TypedDict):
            """Route query to destination."""
            destination: Literal["code_generation", "negative_code_generation"]

        route_chain = (
            route_prompt
            | self.cisco_llm.with_structured_output(RouteQuery)
            | itemgetter("destination")
        )

        self.route = route_chain.invoke(self.user_query)

        self.logger.info(f"---Route:: {self.route} ---")
        

    def _context_summary(self, state: _GraphState) -> _GraphState:
        """_summary_

        Args:
            state (GraphState): _description_

        Returns:
            GraphState: _description_
        """
        context = state["context"]
        
        context_summary_chain = self.context_summary_prompt | self.cisco_llm.with_structured_output(self.summary_struc, include_raw=True) | self._parse_output

        new_context = context_summary_chain.invoke(
            {"context": context}
        )

        new_context = new_context.summary
        
        self.logger.info(f"---CONTEXT:: {new_context} ---")

        state.update({
                "context" : new_context
            })
        return state


    def _code_gen(self, state: _GraphState) -> _GraphState:
        """
            Generates code
        Args:
            state (_GraphState): the current state of the graph

        Returns:
            state (_GraphState): new key added to the state
        """

        code_gen_chain = self.code_gen_prompt | self.cisco_llm.with_structured_output(self.code_struc, include_raw=True) | self._parse_output
        neg_code_gen_chain = self.neg_code_gen_prompt | self.cisco_llm.with_structured_output(self.code_struc, include_raw=True) | self._parse_output


        if self.route == 'negative_code_generation':
            chain = neg_code_gen_chain
            self.code_file_name = self.code_file_name_path + self.user_endpoint.split("/")[-1] + "_neg.py"
        elif self.route == 'code_generation':
            self.code_file_name = self.code_file_name_path + self.user_endpoint.split("/")[-1] + ".py"
            chain = code_gen_chain
        else:
            raise Exception("Failed to pick route")
        
        '''
        self.logger.info("---PAUSING---")
        time.sleep(5)
        self.logger.info("---RESUMING---")'''
        
        messages = state["messages"]
        iterations = state["iterations"]
        context = state["context"]

        # Solution
        self.logger.info("---GENERATING CODE---")
        try:
            code_solution = chain.invoke(
                {"context": self.context, 
                "list_env_var": ["MERAKI_API_KEY", "ORGANIZATION_ID"] ,
                "test_framework": self.framework, 
                "user_query" : self.user_query, 
                "endpoint" : self.user_endpoint ,
                "messages": [self.user_query]}
            )
        except Exception as e:
            self.logger.error("---CODE GENERATE: FAILED---", e)
        messages += [
            (
                "assistant",
                f"{code_solution.prefix} \n Code: {code_solution.code}",
            )
        ]

        # Increment
        iterations = iterations + 1
        code_solution = code_solution.code

        stored_code = code_solution
        stored_code = "#USER QUERY: " + self.user_query + "\n"+ "#USER ENDPOINT: " + self.user_endpoint + "\n\n" + stored_code

        
        
        with open(self.code_file_name, "w") as f:
            f.write(stored_code)

        self.logger.info("---CODE GENERATED SUCCESSFULLY---")

        state.update({
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error" : [False,'']
            })
        return state


    def _code_lint(self, state: _GraphState) -> _GraphState:
        """
        Uses Ruff for static check

        Args:
            state (_GraphState): _description_

        Returns:
            _GraphState: _description_
        """
        messages = state["messages"]
        self.logger.info("---LINTING CODE WITH RUFF---")
        try:
            lint_results = subprocess.run([sys.executable, '-m', 'ruff', 'check', '--fix' , self.code_file_name], capture_output=True, text=True)

            output = lint_results.stdout
        except Exception as e:
            self.logger.warning("---CODE BLOCK LINT RUN: FAILED---", e)


        if lint_results.returncode == 0:
            return state
        else:
            self.logger.warning("---CODE BLOCK LINT: ERROR FOUND---")

            if self.show_output:
                self.logger.info(output)

            output_message = [("user", f"Your solution results: {output}")]

            messages += output_message
            state.update({
                "messages": messages,
                "error": [True,output],
                "evaluate" : [output]
            })
            return state


    def _code_check(self, state: _GraphState):
        """
        Check code, code is ran for errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """


        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]
        self.logger.info("---CODE BLOCK RUN---")
        self.logger.info(f"---Iterations: {iterations}---")

        try:
            results = subprocess.run([sys.executable, '-m', 'pytest', '-v' , self.code_file_name], capture_output=True, text=True)
            output = results.stdout
            error = results.stderr
        except Exception as e:
            self.logger.error("---CODE BLOCK RUN: FAILED---", e)

        if results.returncode == 0:
            pass
        else:
            self.logger.warning("---CODE BLOCK CHECK: FAILED---")

            output = output + '\n' + error

            if self.show_output:
                self.logger.info(output)

            output_message = [("user", f"Your solution results: {output}")]
            #error_message = [("user", f"Your solution failed the code execution test: {error}")]
            #messages += error_message
            messages += output_message
            state.update({
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": [True,output],
            })
            return state

        # No errors
        self.logger.info("---NO CODE TEST FAILURES---")
        state.update({
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": [False],
        })
        return state


    def _create_db_doc(self,state: _GraphState) -> dict:

        current_time = datetime.now()

        if state["error"][self.ERRORED_POS]:
            error_mes = state["error"][self.ERROR_MESS_POS]
        else:
            error_mes = None


        #add context
        doc = {
            "prompt" : self.user_query,
            "system_prompt_code_generation" : str(self.code_gen_prompt),
            "system_prompt_code_fix": str(self.code_fix_prompt),
            "context":state["context"],
            "endpoint" : self.user_endpoint,
            "generated_code" : state["generation"],
            "errored": state["error"][self.ERRORED_POS],
            "error" : error_mes,
            "iterations" : state["iterations"],
            "mod_timestamp": int(current_time.strftime("%H%M%S")),
            "mod_datestamp": current_time.strftime("%Y-%m-%d")
        }

        return(doc)


    def _code_store(self,state: _GraphState) -> None:
        """
        stores the prompt and code in database

        Args:
            state (_GraphState): _description_
        """

        try:
            db_client = pymongo.MongoClient(self.db_url)

            db_list = db_client.list_database_names()

            if self.db_name not in db_list:
                raise Exception(f"{self.db_name} was not found in list of databases in {db_client}.")

            MAGICdb = db_client[self.db_name]
            col_list = MAGICdb.list_collection_names()

            if self.collection_name not in col_list:
                raise Exception(f"{self.collection_name} was not found in the Database collections, {col_list}")

            MAGICcol = MAGICdb[self.collection_name]

            self.logger.info("---STORING---")
            MAGICcol.insert_one(self._create_db_doc(state))

            try:
                self.logger.info("---STORING---")
                MAGICcol.insert_one(self._create_db_doc(state))
            except Exception as e:
                self.logger.error("Failed to insert Document: ",e)

        except Exception as e:
            self.logger.error("Failed to load DataBase: ",e)


    def _code_evaluate(self, state: _GraphState) -> _GraphState:
        """_summary_

        Args:
            state (_GraphState): _description_

        Returns:
            _GraphState: _description_
        """

        code_evaluate_chain = self.code_evaluation_prompt | self.cisco_llm

        error = state["error"]
        messages = state["messages"]

        code_evaluation = code_evaluate_chain.invoke(
            {"error": error[self.ERROR_MESS_POS], "context": state["context"] , "messages": messages}
        )

        self.logger.info("---Code Evaluation---")
        self.logger.info(code_evaluation)
        self.logger.info("---Code Evaluation End---")

        state.update({
            "messages": messages,
            "evaluate": code_evaluation
            })
        return state


    def _code_fix(self, state: _GraphState) -> _GraphState:
        """
            Generates fixes for code
        Args:
            state (_GraphState): the current state of the graph

        Returns:
            state (_GraphState): new key added to the state
        """
        self.logger.info("---CODE FIX---")
        code_fix_chain = self.code_fix_prompt | self.cisco_llm.with_structured_output(self.code_struc, include_raw=True) | self._parse_output
        neg_code_fix_chain = self.neg_code_fix_prompt | self.cisco_llm.with_structured_output(self.code_struc, include_raw=True) | self._parse_output
        
        if self.route == 'negative_code_generation':
            chain = neg_code_fix_chain
        elif self.route == 'code_generation':
            chain = code_fix_chain
        else:
            raise Exception("Failed to pick route")
        
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        code_evaluation = state["evaluate"]

        # Solution
        
        code_fix_solution = chain.invoke(
            {"error": error[self.ERROR_MESS_POS], "evaluation" : code_evaluation , "messages": messages}
        )

        messages += [
            (
                "assistant",
                f"{code_fix_solution.prefix} \n Code: {code_fix_solution.code}",
            )
        ]

        # Increment
        iterations = iterations + 1
        code_fix_solution = code_fix_solution.code

        stored_code = code_fix_solution
        stored_code = "#USER QUERY: " + self.user_query + "\n"+ "#USER ENDPOINT: " + self.user_endpoint + "\n\n" + stored_code

        with open(self.code_file_name, "w") as f:
            f.write(stored_code)
        self.logger.info("---CODE FIX COMPLETE---")
        state.update({
            "generation": code_fix_solution,
            "messages": messages,
            "iterations": iterations
            })
        return state


    def _decide_to_lint(self, state: _GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]

        if error[self.ERRORED_POS] == False:
            if self.show_iterations:
                self.logger.info(f"---Iterations: {iterations}---")
            self.logger.info("---DECISION: Passed Lint---")
            return "pass_on"
        else:
            self.logger.info("---DECISION: Failed Lint---")
            return "re-gen"


    def _decide_to_finish(self, state: _GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]

        if error[self.ERRORED_POS] == False:
            if self.show_iterations:
                self.logger.info(f"---Iterations: {iterations}---")
            self.logger.info("---DECISION: Successful END---")
            return "end"
        elif iterations >= self.max_iterations:

            self.logger.info("---DECISION: FINISH Max Reached---")
            return "end"
        else:
            self.logger.info("---DECISION: RE-TRY SOLUTION---")
            return "re-gen"


    def _code_report(self, state: _GraphState):
        """
        reports more details to log

        Args:
            state (_GraphState): _description_
        """
        iterations = state["iterations"]
        if self.show_iterations:
            self.logger.info(f"---Iterations: {iterations}---")


    def _get_context(self, endpoint:str, test_query:str) -> str:
        """
        Retrieve  context from the API Spec Document

        Args:
            endpoint (str): endpoint url which will be tested
            query (str): test case used to generate tests

        Returns:
            str: Retrieved Api document section
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        document_path  = self.spec_sheet
        vectorDbPath = self.chroma_db_dir
        db_collective_name = "Records"

        ollama_emb = OllamaEmbeddings(
            model="llama3.2"
        )

        aug_endpoint = self.chunks_dir + endpoint.replace("/", "_").replace("{", "_").replace("}", "_")
        possible_endpoints = [aug_endpoint+"_get.json", aug_endpoint +"_put.json",aug_endpoint + "_post.json", aug_endpoint+"_delete.json"]

        filter_dict = {"source": {"$in" : possible_endpoints}}
        client = chromadb.PersistentClient(path=vectorDbPath)

        vector_store = Chroma(collection_name=db_collective_name, client=client,embedding_function=ollama_emb,create_collection_if_not_exists=False) 
        
        try:
            response = vector_store.similarity_search(query=test_query,k=1, filter = filter_dict)
            response:str =response[0].page_content
        except Exception as e:
            response = "Failed To Retrieval"
            self.logger.exception(e)

        return response


    def create_graph_workflow(self):
        workflow = StateGraph(self._GraphState)

        # Define the nodes
        workflow.add_node("summarize",self._context_summary)
        workflow.add_node("generate", self._code_gen)  # generation solution
        workflow.add_node("lint", self._code_lint)
        workflow.add_node("check_code", self._code_check)  # check code
        workflow.add_node("eval", self._code_evaluate)
        workflow.add_node("make_fix", self._code_fix)
        #workflow.add_node("reflect", reflect)  # reflect
        workflow.add_node("store", self._code_store)

        # Build graph
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", "lint")
        workflow.add_conditional_edges(
            "lint",
            self._decide_to_lint,
            {
                "pass_on": "check_code",
                "re-gen": "make_fix",
            },
        )
        workflow.add_edge("check_code", "store" )
        workflow.add_conditional_edges(
            "check_code",
            self._decide_to_finish,
            {
                "end": END,
                "re-gen": "eval",
            },
        )
        workflow.add_edge("eval", "make_fix")
        workflow.add_edge("make_fix", "check_code")

        return workflow.compile()

    def run_app(self, user_endpoint:str, user_query:str) -> dict:
        """
        main entry program

        Args:
            user_endpoint (str): _description_
            user_query (str): _description_

        Returns:
            dict: _description_
        """
        self.user_endpoint = user_endpoint
        self.user_query = user_query
        
        self._intent()
        self._get_examples()
        self._create_prompt()
        
        context : str = self._get_context(user_endpoint, user_query)
        self.logger.info("\n\n--- Starting MAGIC ---")

        self.context = context

        self.logger.info("\n--- MAGIC ---")

        self.code_file_name = self.code_file_name_path + self.user_endpoint.split("/")[-1] + ".py"
        

        app = self.create_graph_workflow()

        solution:dict = app.invoke({
            "context": context,
            "list_env_var": ["MERAKI_API_KEY", "ORGANIZATION_ID"],
            "test_framework": self.framework,
            "user_query" : user_query,
            "endpoint" : user_endpoint,
            "messages": [user_query],
            "iterations": 0,
            "error": ""
        })
        self.logger.info("--- Magic Complete ---")

        solution["endpoint"] = self.user_endpoint
        solution["code_filename"] = self.code_file_name
        #print(f"Iterations: {solution['iterations']}")
        #print(f"Endpoint: {solution['endpoint']}  {self.user_endpoint}")
        return(solution)
