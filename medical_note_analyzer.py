from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from typing import List
from pydantic import BaseModel, Field
import json

from key import openai_key
import os
os.environ['OPENAI_API_KEY'] = openai_key

# Define the output structure for medication analysis
class MedicationAnalysis(BaseModel):
    medication_name: str = Field(description="Name of the medication")
    potential_interactions: List[str] = Field(description="List of potential drug interactions")
    lifestyle_impacts: List[str] = Field(description="How this medication affects daily life")
    monitoring_needs: List[str] = Field(description="What needs to be monitored while on this med")

class PatientLifestyleImpact(BaseModel):
    daily_routine_changes: List[str] = Field(description="Required changes to daily routine")
    dietary_restrictions: List[str] = Field(description="Required dietary changes")
    activity_modifications: List[str] = Field(description="Physical activity adjustments")
    
def create_medication_analyzer():
    # Create parser and prompt template
    parser = PydanticOutputParser(pydantic_object=MedicationAnalysis)
    
    prompt = PromptTemplate(
        template="""
        Analyze the following medication from a clinical note and provide detailed information about its impacts:
        
        Medication: {medication}
        
        {format_instructions}
        """,
        input_variables=["medication"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create chain
    llm = OpenAI(temperature=0.6)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain, parser

def create_lifestyle_analyzer():
    parser = PydanticOutputParser(pydantic_object=PatientLifestyleImpact)
    
    prompt = PromptTemplate(
        template="""
        Given the patient's condition and medications, analyze the lifestyle impacts:
        
        Condition: {condition}
        Medications: {medications}
        Current Lifestyle: {lifestyle}
        
        {format_instructions}
        """,
        input_variables=["condition", "medications", "lifestyle"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = OpenAI(temperature=0.6)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain, parser

def analyze_clinical_note(note: str):
    """
    Main function to analyze a clinical note and provide comprehensive insights
    """
    # Extract medications from note using OpenAI
    llm = OpenAI(temperature=0)
    extraction_prompt = PromptTemplate(
        template="Extract all medications from this clinical note: {note}",
        input_variables=["note"]
    )
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    medications = extraction_chain.run(note=note).split('\n')
    
    # Analyze each medication
    med_analyzer, med_parser = create_medication_analyzer()
    medication_analyses = []
    
    for med in medications:
        if med.strip():
            result = med_analyzer.run(medication=med)
            parsed_result = med_parser.parse(result)
            medication_analyses.append(parsed_result)
    
    # Analyze lifestyle impacts
    lifestyle_analyzer, lifestyle_parser = create_lifestyle_analyzer()
    lifestyle_result = lifestyle_analyzer.run(
        condition="",  # Extract from note
        medications=", ".join(medications),
        lifestyle=""  # Extract from note
    )
    lifestyle_analysis = lifestyle_parser.parse(lifestyle_result)
    
    # Combine results
    final_analysis = {
        "medications": [json.loads(med.json()) for med in medication_analyses],
        "lifestyle_impacts": json.loads(lifestyle_analysis.json())
    }
    
    return final_analysis