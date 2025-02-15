# prepares quiz questions based on the uploaded files

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.program.evaporate.df import DFRowsProgram
from llama_index.program.openai import OpenAIPydanticProgram
from global_settings import INDEX_STORAGE, QUIZ_SIZE, QUIZ_FILE
import pandas as pd
import sys
import traceback

def build_quiz(topic):
    print(f"Starting quiz generation for topic: {topic}")
    print(f"Targeting QUIZ_SIZE: {QUIZ_SIZE}")

    try:
        # Create initial DataFrame
        df = pd.DataFrame(
            {
                "Question_no": pd.Series(dtype="int"),
                "Question_text": pd.Series(dtype="str"),
                "Option1": pd.Series(dtype="str"),
                "Option2": pd.Series(dtype="str"),
                "Option3": pd.Series(dtype="str"),
                "Option4": pd.Series(dtype="str"),
                "Correct_answer": pd.Series(dtype="str"),
                "Rationale": pd.Series(dtype="str"),
            }
        )

        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)

        # Load vector index
        vector_index = load_index_from_storage(
            storage_context, index_id="vector"
        )

        # Create query engine
        query_engine = vector_index.as_query_engine()

        # Retrieve context from the index to inform quiz generation
        context_query = f"Summarize the key points and main themes related to {topic}"
        context_response = query_engine.query(context_query)

        # Construct quiz generation query
        query = (
            f"Generate exactly {QUIZ_SIZE} different multiple-choice quiz questions about {topic}. "
            f"Context: {context_response}\n\n"
            "Requirements for EACH question:"
            "- Directly relate to the provided context"
            "- Have 4 multiple-choice answer options"
            "- Mark the correct answer"
            "- Provide a detailed rationale"
            "- Cover different aspects of the topic"
            f"Ensure you generate EXACTLY {QUIZ_SIZE} questions."
        )

        # Prepare DataFrame rows program
        df_rows_program = DFRowsProgram.from_defaults(
            pydantic_program_cls=OpenAIPydanticProgram, 
            df=df
        )

        # Execute query
        response = query_engine.query(query)

        # Process response
        result_obj = df_rows_program(input_str=str(response))
        
        # Convert to DataFrame
        new_df = result_obj.to_df(existing_df=df)

        # Verify number of questions
        print(f"Generated questions: {len(new_df)}")
        if len(new_df) != QUIZ_SIZE:
            print(f"WARNING: Generated {len(new_df)} questions instead of {QUIZ_SIZE}")

        # Ensure Question_no is set correctly
        new_df['Question_no'] = range(1, len(new_df) + 1)

        # Fallback mechanism if needed
        if len(new_df) != QUIZ_SIZE:
            print(f"Attempting to regenerate questions to match {QUIZ_SIZE}")
            
            if len(new_df) < QUIZ_SIZE:
                # Pad with placeholder questions
                padding_df = pd.DataFrame([{
                    'Question_text': 'Placeholder question',
                    'Option1': 'Option A',
                    'Option2': 'Option B',
                    'Option3': 'Option C',
                    'Option4': 'Option D',
                    'Correct_answer': 'Option A',
                    'Rationale': 'Placeholder rationale'
                }] * (QUIZ_SIZE - len(new_df)))
                new_df = pd.concat([new_df, padding_df], ignore_index=True)
            else:
                # Trim to exact QUIZ_SIZE
                new_df = new_df.head(QUIZ_SIZE)
            
            # Reset question numbers
            new_df['Question_no'] = range(1, len(new_df) + 1)

        # Save to CSV
        new_df.to_csv(QUIZ_FILE, index=False)

        print("Quiz generation completed successfully")
        print("\nGenerated Quiz:")
        print(new_df)
        return new_df

    except Exception as e:
        print("ERROR: An exception occurred during quiz generation")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        raise

# Main block for direct testing
if __name__ == "__main__":
    try:
        # Prompt user to input the topic
        topic = input("Enter the topic for the quiz: ")
        result = build_quiz(topic)
    except Exception as e:
        print("Quiz generation failed in main block")
        traceback.print_exc()
        sys.exit(1)

