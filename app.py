import streamlit as st
import os
from rag_pipeline import load_vectorstore, setup_rag_pipeline, MY_GOOGLE_API_KEY

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")
st.title("🩺 Medical RAG QA System")

api_key = MY_GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def initialize_system():
    try:
        vectorstore = load_vectorstore("vectorstore")
        chain = setup_rag_pipeline(vectorstore, api_key)
        return chain
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        return None

qa_chain = initialize_system()


# Sidebar navigation
section = st.sidebar.radio("Select Section", ["QA", "Evaluation"])

if section == "QA":
    if qa_chain is not None:
        st.success("✅ System Ready!")
        st.subheader("Ask a Medical Question")
        query = st.text_input("Enter your question:")
        if query:
            with st.spinner("Processing..."):
                try:
                    response = qa_chain.invoke({"query": query})
                    st.subheader("Answer:")
                    st.write(response["result"])
                    st.subheader("Sources:")
                    for i, doc in enumerate(response["source_documents"]):
                        with st.expander(f"Source {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Failed to initialize system. Run: python preprocess_data.py")

elif section == "Evaluation":
    st.header("Single Question Evaluation")
    st.write("Select a question and run evaluation for it.")

    questions = [
        "What are the symptoms of allergic rhinitis?",
        "How to treat nasal allergies effectively?",
        "What is allergic rhinitis and its common triggers?",
        "Describe the procedure for laparoscopic gastric bypass.",
        "What are the complications of gastric bypass surgery?",
        "How much weight can be lost with laparoscopic gastric bypass?",
        "What is the recovery time for gastric bypass?",
        "What was found in the 2-D Echocardiogram?",
        "What are the signs of left atrial enlargement?",
        "Explain mitral regurgitation and its causes.",
        "What does an ejection fraction of 70% indicate?",
        "How is aortic valve stenosis treated?",
        "How is chronic kidney disease managed?",
        "What are the stages of kidney disease?",
        "How are electrolyte imbalances treated in kidney disease?",
        "What are the symptoms of migraine headaches?",
        "How is Bell's palsy diagnosed and treated?",
        "What causes peripheral neuropathy?",
        "What is the treatment for knee pain?",
        "How to manage lower back pain?",
        "What causes osteoarthritis?",
        "What is GERD and how is it treated?",
        "What causes peptic ulcers?",
        "How are gallstones managed?",
        "What is COPD and its risk factors?",
        "How is asthma managed?",
        "What causes shortness of breath?",
        "How is diabetes managed?",
        "What is thyroid disease?",
        "How is hypertension treated?",
        "What are the vital signs and their normal ranges?",
        "How is blood pressure measured?",
        "What does a physical examination include?",
    ]

    selected_question = st.selectbox("Select a question:", questions)
    if st.button("Run Evaluation for Selected Question"):
        if qa_chain is not None:
            with st.spinner("Evaluating..."):
                try:
                    response = qa_chain.invoke({"query": selected_question})
                    st.success("Evaluation completed!")
                    st.subheader("Question:")
                    st.write(selected_question)
                    st.subheader("Answer:")
                    st.write(response["result"])
                    st.subheader("Sources:")
                    for i, doc in enumerate(response["source_documents"]):
                        with st.expander(f"Source {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Failed to initialize system. Run: python preprocess_data.py")