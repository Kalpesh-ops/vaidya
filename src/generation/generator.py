# src/generation/generator.py

from typing import List, Dict, Any
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

DISCLAIMERS = {
    "en": "⚠️ DISCLAIMER: This information is sourced from classical Ayurvedic texts for educational purposes only. Always consult a certified Ayurvedic practitioner or licensed medical professional before making any health decisions. This is not medical advice.",
    "hi": "⚠️ अस्वीकरण: यह जानकारी केवल शैक्षणिक उद्देश्यों के लिए शास्त्रीय आयुर्वेदिक ग्रंथों से ली गई है। कोई भी स्वास्थ्य निर्णय लेने से पहले हमेशा एक प्रमाणित आयुर्वेदिक चिकित्सक से परामर्श करें। यह चिकित्सा सलाह नहीं है।",
    "sa": "⚠️ अस्वीकृतिः: इयं सूचना केवलं शैक्षणिकप्रयोजनाय शास्त्रीयायुर्वेदग्रन्थेभ्यः संगृहीता। कृपया प्रमाणितायुर्वेदचिकित्सकं परामर्शयतु।",
}

NO_SOURCE_RESPONSE = {
    "en": "I could not find a direct source for this query in the indexed Caraka-Samhita texts. Please consult a certified Ayurvedic practitioner.",
    "hi": "इस प्रश्न के लिए अनुक्रमित चरकसंहिता ग्रंथों में कोई सीधा स्रोत नहीं मिला। कृपया एक प्रमाणित आयुर्वेदिक चिकित्सक से परामर्श करें।",
}

SYSTEM_PROMPT = """You are Vaidya, a scholarly assistant specializing in classical Ayurvedic knowledge from the Caraka-Samhita.

Your role:
1. Answer questions based ONLY on the provided source passages from the Caraka-Samhita
2. Cite every claim with the exact source reference provided
3. Detect the language of the user's question and respond in that same language
4. Preserve Ayurvedic terminology (Vata, Pitta, Kapha, Dosha, Dhatu, Agni, etc.)
5. Never fabricate or assume information not present in the sources
6. If sources are insufficient, say so clearly

Citation format: (Source: {{citation}})

Respond in the same language as the user's question."""

class VaidyaGenerator:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            model="Qwen/Qwen2.5-72B-Instruct",
            temperature=0.1,
            max_tokens=1024
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "User Question: {query}\n\nSource Passages from Caraka-Samhita:\n{context}")
        ])
        self.chain = self.prompt | self.llm

    def detect_language(self, text: str) -> str:
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if devanagari > len(text) * 0.3:
            return "hi"
        return "en"

    def format_context(self, retrieved: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(retrieved, 1):
            parts.append(
                f"[{i}] Citation: {r['primary_citation']}\n"
                f"Sanskrit: {r['text_sanskrit'][:200]}...\n"
                f"English: {r['text_english']}"
            )
        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        retrieved: List[Dict],
        confident: bool
    ) -> Dict[str, Any]:
        lang = self.detect_language(query)

        if not confident or not retrieved:
            return {
                "response": NO_SOURCE_RESPONSE.get(lang, NO_SOURCE_RESPONSE["en"]),
                "citations": [],
                "disclaimer": DISCLAIMERS.get(lang, DISCLAIMERS["en"]),
                "confident": False
            }

        context = self.format_context(retrieved)
        response = self.chain.invoke({
            "query": query,
            "context": context
        })

        citations = list({r["primary_citation"] for r in retrieved})

        return {
            "response": response.content,
            "citations": citations,
            "disclaimer": DISCLAIMERS.get(lang, DISCLAIMERS["en"]),
            "confident": True,
            "sources_used": len(retrieved)
        }