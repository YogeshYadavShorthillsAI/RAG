import json

def generate_test_cases():
    test_cases = [
    {
        "TEST CASE ID": "TC_001",
        "SECTION": "Human History",
        "SUB-SECTION": "Sexual Division of Labor",
        "TEST CASE TITLE": "Verify the concept of Sexual Division of Labor (SDL)",
        "TEST DESCRIPTION": "Ensure that the system correctly identifies and explains the concept of SDL and its differences among human hunter-gatherer societies compared to other species.",
        "PRECONDITIONS": "Relevant historical context and literature available in the database.",
        "TEST DATA": {
            "context": "Sexual division of labour (SDL) is the delegation of different tasks between the male and female members of a species. Among human hunter-gatherer societies, males and females are responsible for the acquisition of different types of foods and shared them with each other for a mutual or familial benefit.",
            "question": "What is the sexual division of labor (SDL) and how does it differ among human hunter-gatherer societies compared to other species?"
        },
        "TEST STEPS": [
            "Submit the question and context to the system.",
            "Retrieve the generated answer from the system.",
            "Compare the generated answer with the expected answer."
        ],
        "EXPECTED RESULT": "The system should provide an explanation of SDL, detailing its role in human hunter-gatherer societies and highlighting the unique combination of food acquisition and sharing.",
        "ACTUAL RESULT": "as expected",
        "STATUS": "Pass"
    },
    {
        "TEST CASE ID": "TC_002",
        "SECTION": "Human History",
        "SUB-SECTION": "Hunter-Gatherer Studies",
        "TEST CASE TITLE": "Verify system understanding of Hadza population studies",
        "TEST DESCRIPTION": "Ensure that the system accurately describes the contribution of Hadza studies in understanding the sexual division of labor and its relevance in modern society.",
        "PRECONDITIONS": "System has knowledge of anthropological studies on hunter-gatherer populations.",
        "TEST DATA": {
            "context": "The few remaining hunter-gatherer populations in the world serve as evolutionary models that can help explain the origin of the sexual division of labour. Many studies on the sexual division of labour have been conducted on hunter-gatherer populations, such as the Hadza, a hunter-gatherer population of Tanzania.",
            "question": "How do studies on hunter-gatherer populations like the Hadza of Tanzania contribute to our understanding of the sexual division of labor, and how does this division of labor manifest in modern-day society?"
        },
        "TEST STEPS": [
            "Submit the question and context to the system.",
            "Retrieve the generated answer from the system.",
            "Compare the generated answer with the expected answer."
        ],
        "EXPECTED RESULT": "The system should describe how Hadza studies provide insights into gendered roles in hunter-gatherer societies and draw parallels to modern-day occupational patterns.",
        "ACTUAL RESULT": "as expected",
        "STATUS": "Pass"
    },
    {
        "TEST CASE ID": "TC_003",
        "SECTION": "Human History",
        "SUB-SECTION": "Life History Theory",
        "TEST CASE TITLE": "Verify system's understanding of life history theory in relation to reproductive investment",
        "TEST DESCRIPTION": "Ensure that the system correctly explains how life history theory accounts for different reproductive investment strategies between males and females.",
        "PRECONDITIONS": "System has access to evolutionary biology principles.",
        "TEST DATA": {
            "context": "Women have the option of investing resources either to provision children or to have additional offspring. According to life history theory, males and females monitor costs and benefits of each alternative to maximize reproductive fitness; however, trade-off differences do exist between sexes. Females are likely to benefit most from parental care effort because they are certain which offspring are theirs and have relatively few reproductive opportunities, each of which is relatively costly.",
            "question": "How does life history theory explain the different investment strategies between males and females regarding offspring provision and reproduction?"
        },
        "TEST STEPS": [
            "Submit the question and context to the system.",
            "Retrieve the generated answer from the system.",
            "Compare the generated answer with the expected answer."
        ],
        "EXPECTED RESULT": "The system should explain how life history theory highlights differences in reproductive strategies, emphasizing female certainty in offspring and the higher cost of reproduction for females.",
        "ACTUAL RESULT": "as expected",
        "STATUS": "Pass"
    },
    {
        "TEST CASE ID": "TC_004",
        "SECTION": "Human History",
        "SUB-SECTION": "Gendered Roles in Hunter-Gatherers",
        "TEST CASE TITLE": "Verify system's knowledge of historical perspectives on gendered roles in hunter-gatherer societies",
        "TEST DESCRIPTION": "Ensure that the system accurately identifies the anthropologists who coined the \"Man the Hunter, Woman the Gatherer\" model and its historical significance.",
        "PRECONDITIONS": "System has knowledge of anthropological literature on gender roles in early societies.",
        "TEST DATA": {
            "context": "From the 1970s onward, the dominant paleontological perspective of gendered roles in hunter-gatherer societies was of a model termed \"Man the Hunter, Woman the Gatherer\"; coined by anthropologists Richard Borshay Lee and Irven DeVore in 1968, it argued, based on evidence now thought to be incomplete, that contemporary foragers displayed a clear division of labor between women and men.",
            "question": "What is the historical perspective on gendered roles in hunter-gatherer societies, as highlighted in the text, and who were the anthropologists who coined this perspective?"
        },
        "TEST STEPS": [
            "Submit the question and context to the system.",
            "Retrieve the generated answer from the system.",
            "Compare the generated answer with the expected answer."
        ],
        "EXPECTED RESULT": "The system should identify Richard Borshay Lee and Irven DeVore as the anthropologists who coined the \"Man the Hunter, Woman the Gatherer\" model and explain its historical significance.",
        "ACTUAL RESULT": "as expected",
        "STATUS": "Pass"
    }
]

    with open("evaluation_unittest_cases.json", "w") as file:
        json.dump(test_cases, file, indent=4)

    print("Test cases JSON file has been created successfully.")

if __name__ == "__main__":
    generate_test_cases()
