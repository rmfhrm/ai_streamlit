from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from config import answer_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 1. llm 셋팅 함수// llm이라는 함수를 통해서 retrun받을 수있다.
def get_llm(model="gpt-4o"):
    llm=ChatOpenAI(model=model)
    return llm

# 2. dictionary 체이닝 함수
def get_dictionary_chain():
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경이 필요없을 경우, 사용자의 질문을 변경하지 않아도 됩니다.
    사전:{dictionary}
    질문:{{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser() # <= 변경된 질문을 리턴한다
    return dictionary_chain

# 3. retriever 함수
def get_retriever():
    # pinecone 데이터 임베딩
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    # pinecone index 데이터 로드
    index_name = 'tax-index'
    # 기존에 있는 데이터를 뽑아오는 작업
    vectorstore=PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    # retriever 셋팅
    retriever=vectorstore.as_retriever(search_kwargs={'k':3})
    return retriever # retriever만 있다면 내가 원하는 자료 db에서 가져올 수 있다.

# 4. QA 체인 함수
def get_history_retriever():
    llm=get_llm()
    retriever=get_retriever()

    # prompt 셋팅
    #prompt = hub.pull("rlm/rag-prompt")     # 허깅페이스에서 제공해줌
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),        # system과 human 사이에 ? 이거 왜쓰는가? 기억하기 위해서?
            ("human", "{input}"),                       # 지난 질문을 기억 문맥에 맞게끔 질문에 답을 함 =messageplaceholder
        ]
    )
    # chain 연결 
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()

    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,   #config에 만든 few shot 사용
    )
    history_aware_retriever = get_history_retriever()

    #주 prompt
    system_prompt = (
        "당신은 소득세법 전문가 입니다. 사용자의 소득세법에 관한 질문에 답변해 주세요."
        "아래에 제공된 문서를 활용해서 답변해 주시고,"
        "답변을 알 수 없다면 모른다고 답변해 주세요. "
        "답변을 제공할 때 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해 주시고, "
        "2~3 문장정도의 짧은 내용의 답변을 원합니다. "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),    #과거에 대화를 나누었던  history/ ㄴ문맥에 전체의 최종응답을 하기위해 전체다 
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)    #chain 2개 연결??
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    #RAG체인을 연결해줌
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain # == qa_chain이랑 같데,, 왜지??
    

# 5. AI 관련 데이터 셋팅 함수
def get_ai_messages(user_message):
    dictionary_chain=get_dictionary_chain()
    qa_chain = get_qa_chain()

    tax_chain = {"input": dictionary_chain} | qa_chain
    ai_message = tax_chain.stream(
        {
            "question":user_message,
        },
        config={
        "configurable": {"session_id": "abc123"}
    },
    )
    
    # 결과 반환
    return ai_message
    # ui 에서 ~ 따르면 나오는 이유 내가  config 안 few sthot에서 설정해 두었기 때문에