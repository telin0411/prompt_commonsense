import os
import json
import glob
import pprint



def extract_eqasc(json_file):
    extract_data = []
    data = json.load(open(json_file))

    for datum in data:
        question = datum["question"]
        answerKey = datum["answerKey"]
        fact1 = datum["fact1"]
        fact2 = datum["fact2"]
        combinedfact = datum["combinedfact"]
        overlapping_entities = datum["overlapping_entities"]

        # print(answerKey)
        # print(fact1)
        # print(fact2)
        # print(combinedfact)
        # print(overlapping_entities)

        # print(question.keys())
        question_text = question["stem"]
        # print(question_text)

        positive_explanations = []
        negative_explanations = []
        answer = None

        for i in range(len(question["choices"])):
            choice_dict = question["choices"][i]
            label = choice_dict["label"]
            label_text = choice_dict["text"]
            chains = choice_dict["chains"]

            for chain in chains:
                text_1 = chain[0]["text"]
                text_2 = chain[1]["text"]
                combined_explanation_chains = " ".join([text_1, text_2])
                # Positive explanations.
                if label == answerKey:
                    answer = label_text
                    turk_label = chain[2]["turk_label"]["label"]
                    if turk_label == "yes":
                        positive_explanations.append(combined_explanation_chains)
                    else:
                        negative_explanations.append(combined_explanation_chains)
                # Negative explanations.
                else:
                    negative_explanations.append(combined_explanation_chains)

        extract_datum = {
            "statement": question_text,
            "positive_explanations": positive_explanations,
            "negative_explanations": negative_explanations,
            "answers": answer,
            "data_resource": "eQASC",
        }
        extract_data.append(extract_datum)

    return extract_data


def extract_from_jsons(data_root, args=None):
    extract_data = []

    json_files = glob.glob(os.path.join(data_root, "*.json"))
    for json_file in json_files:
        print("Processing: {}".format(json_file))
        extract_data += extract_eqasc(json_file)
    
    eqasc_out_f = "eQASC_explanations.jsonl"
    with open(eqasc_out_f, "w") as outf:
        for d in extract_data:
            outf.write(json.dumps(d)+"\n")
        pass
    print("Saving eQASC explanations data to: {}".format(eqasc_out_f))

    return None


if __name__ == "__main__":
    data_root = "datasets/eqasc"
    extract_from_jsons(data_root)
