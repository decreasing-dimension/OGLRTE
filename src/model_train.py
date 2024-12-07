import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer
import transformers
import params
from model import OGLRTE, loss_fn, save_ckp
from load_data import data_loader

llm_model_path = params.LLM_MODEL_PATH
intstruction = "Given the following ontology, examples and sentence, please extract the triples from the sentence and according to the relations in the ontology. In the output, only include the triples in the given output format."
 
tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=llm_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

val_targets = []
val_outputs = []
triple_outputs = []
final_all = [0, 0, 0]
hallucination = [0, 0]

def cmp(str1, str2):
    n = len(str1)
    m = len(str2)
    if n < m:
        str1, str2 = str2, str1
        n, m = m, n
    if m == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[j - 1] == str2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / m

def trans2triple(outputs, ontology, data):
    outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    triple_output = []
    triple_num = 0
    tp = 0
    fp = 0
    fn = 0
    sh = 0
    oh = 0
    for id, output in enumerate(outputs):
        input1 = "Ontology Concepts: "
        input2 = "Ontology Relations: "
        predict = (np.array(output) >= 0.5).astype(int)
        ent_dict = {}
        predict_list = [0 for i in range(len(predict))]
        data_triples = data["output"][id].split("\n")
        if sum(predict) != 0:
            for i, val in enumerate(predict):
                if val:
                    if ontology["ontology"][i]["sub"] not in ent_dict and ontology["ontology"][i]["sub"] != "": 
                        ent_dict[ontology["ontology"][i]["sub"]] = 1
                        input1 += ontology["ontology"][i]["sub"] + ", "
                    if ontology["ontology"][i]["obj"] not in ent_dict and ontology["ontology"][i]["obj"] != "": 
                        ent_dict[ontology["ontology"][i]["obj"]] = 1
                        input1 += ontology["ontology"][i]["obj"] + ", "
                    input2 += ontology["ontology"][i]["rel"] + "(" + ontology["ontology"][i]["sub"] + ',' + ontology["ontology"][i]["obj"] + "), "
            input1 = input1[:-2]
            input2 = input2[:-2]
            messages = [{"role": "user", "content": intstruction + '\n' + input1 + '\n' + input2 + '\nsentence:' + data["sentence"][id]}]
            result = pipeline(
                messages,
                max_new_tokens=256,
                top_p=1,
                temperature=1,
            )
            result_triples = result[0]["generated_text"][-1]["content"].split("\n")
            triple_num += len(result_triples)
            for triple in result_triples:
                shfla = 0
                ohfla = 0
                alpha = 0.8
                for data_triple in data_triples:
                    if cmp(triple.split('(')[0], data_triple.split('(')[0]) < 0.8:
                        continue
                    res = cmp(" ".join(triple.split('(')[1:]), " ".join(data_triple.split('(')[1:]))
                    if res > alpha:
                        if data_triple.split('(')[0]:
                            if shfla == 0 and cmp(" ".join(triple.split('(')[1:]).split(",")[0], " ".join(data_triple.split('(')[1:]).split(",")[0]) < alpha:
                                sh += 1
                                shfla = 1
                            if ohfla == 0 and cmp(" ".join(triple.split(',')[1:]), " ".join(data_triple.split(',')[1:])) < alpha:
                                oh += 1
                                ohfla = 1
                            tmp = ""
                            for i in range(0, len(data_triple.split('(')) - 1):
                                tmp += data_triple.split('(')[i]
                                if tmp in ontology["node2id"]:
                                    break
                                tmp += "("
                            predict_list[ontology["node2id"][tmp]] = 1
                            tp += 1
                        break
            fp += len(result_triples)
        triple_output.append(predict_list)
        fn += len(data_triples)
    fp = fp - tp
    fn = fn - tp
    return torch.tensor(triple_output, device=device, dtype=torch.float32), tp, fp, fn, sh, oh

def train_model(device, start_epochs, n_epochs, valid_loss_min_input, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path, ontology):
    global val_targets, val_outputs, triple_outputs, final_all, hallucination
    valid_loss_min = valid_loss_min_input
    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('Epoch {}: Training Start'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            label = data['label'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids, ontology)

            optimizer.zero_grad()
            loss = loss_fn(outputs, label)
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, BATCH: {batch_idx}, Training Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('Epoch {}: Training End'.format(epoch))
        print('Epoch {}: Validation Start'.format(epoch))
        model.eval()
        
        part_targets = []
        part_outputs = []
        part_output2 = []
        
        with torch.no_grad():
            all_tp = 0
            all_fp = 0
            all_fn = 0
            all_sh = 0
            all_oh = 0
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                label = data['label'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids, ontology)
                
                output2, tp, fp, fn, sh, oh = trans2triple(outputs, ontology, data)
                all_tp += tp
                all_fp += fp
                all_fn += fn
                all_sh += sh
                all_oh += oh

                loss = loss_fn(outputs, label)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                part_targets.extend(label.cpu().detach().numpy().tolist())
                part_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                part_output2.extend(output2.cpu().detach().numpy().tolist())

            print('Epoch {}: Validation End'.format(epoch))
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            print('Epoch: {} \t Avgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'
                  .format(epoch, train_loss, valid_loss))
            
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) != 0 else 0
            recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) != 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) != 0 else 0
            shull = all_sh / (all_tp + all_fp) if (all_tp + all_fp) != 0 else 0
            ohull = all_oh / (all_tp + all_fn) if (all_tp + all_fn) != 0 else 0
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased from {:.6f} to {:.6f}). Saving model'
                      .format(valid_loss_min, valid_loss))
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss
                val_targets = part_targets
                val_outputs = part_outputs
                final_all = [all_tp, all_fp, all_fn]
                hallucination = [shull, ohull]
        print('Epoch {}  Done\n'.format(epoch))

    return model

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available, '
              f'We will use the GPU: {torch.cuda.get_device_name(0)}.')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    training_loader, validation_loader, ontology = data_loader(params.DATASET_PATH, device)
    
    model = OGLRTE(len(ontology["nodes"]))
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.LEARNING_RATE)
    
    checkpoint_path = './ckp/current_checkpoint.pt'
    best_model = './ckp/best_model.pt'
    trained_model = train_model(device, 1, params.EPOCHS, np.Inf, training_loader, validation_loader, model,
                                optimizer, checkpoint_path, best_model, ontology)
    
    val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
    accuracy = accuracy_score(val_targets, val_predicts)
    f1_score_micro = f1_score(val_targets, val_predicts, average='micro')
    f1_score_macro = f1_score(val_targets, val_predicts, average='macro')
    prec = final_all[0] / (final_all[0] + final_all[1]) if final_all[0] + final_all[1] != 0 else 0
    rec = final_all[0] / (final_all[0] + final_all[2]) if final_all[0] + final_all[2] != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0
    print("Relation filtering results:")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(classification_report(val_targets, val_predicts))
    print("Final results")
    print(f"Predicted Score = {prec}")
    print(f"Recall Score = {rec}")
    print(f"f1 Score = {f1}")
    print(f"SH = {hallucination[0]}")
    print(f"OH = {hallucination[1]}")

