import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script with configurable model and dataset")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="model_type"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        # choices=["UF-P-4", "PRISM", "P-Soups", "AlignX"],
        help="Dataset to evaluate on (UF-P-4, PRISM, P-Soups or AlignX)"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    patha = f"./eval/{args.dataset}_base_eval.jsonl" 
    pathb = f"./eval/{args.dataset}/{args.weights.split('/')[-1]}_eval.jsonl"
    dataA = []
    dataB = []

    with open(patha, 'r') as f:
        for line in f:
            dataA.append(json.loads(line))

    with open(pathb, 'r') as f:
        for line in f:
            dataB.append(json.loads(line))

    dataA.sort(key=lambda x: x['idx'])
    dataB.sort(key=lambda x: x['idx'])

    a_idx_set = set(item['idx'] for item in dataB)
    dataA = [item for item in dataA if item['idx'] in a_idx_set]

    count = 0
    num = 0

    for i, aa in enumerate(dataA):
        if aa["idx"] != dataB[i]["idx"]:
            continue

        chosen_rewards = 0.1 * ((-dataB[i]["nll_loss_all_chosen"]) - (-aa["nll_loss_all_chosen"]))
        rejected_rewards = 0.1 * ((-dataB[i]["nll_loss_all_rejected"]) - (-aa["nll_loss_all_rejected"]))

        if chosen_rewards>rejected_rewards:
            count += 1
            dataB[i]["predict"] = True
        else:
            dataB[i]["predict"] = False
        
        num += 1

    print(num)
    print(count)
    print("acc:", count/num)

if __name__ == "__main__":
    main()