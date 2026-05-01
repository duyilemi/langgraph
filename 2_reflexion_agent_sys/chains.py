import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI  # free
from schema import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGroq(
#     model="llama3-8b-8192",  # cheaper for dev
#     temperature=0
# )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.
Current time: {time}

{instruction}
""",
        ),
        MessagesPlaceholder("messages"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


# Draft
draft_chain = (
    prompt.partial(
        instruction="""
Write a ~150 word answer.
Then critique it (missing + superfluous).
Then provide 1-3 search queries.
"""
    )
    | llm.with_structured_output(AnswerQuestion)
)

# Revise
revise_chain = (
    prompt.partial(
        instruction="""
Revise your answer using the critique and search results.

- Keep under 150 words
- Add citations like [1], [2]
- Add references section
"""
    )
    | llm.with_structured_output(ReviseAnswer)
)