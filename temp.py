import pandas as pd

# data = pd.read_csv('train_data.csv')

# final_data = []
# for i, row in data.iterrows():
#     if len(final_data) > 99:
#         break
#     if (not '_' in row['id']) and (row['label'] < 3):
#         new_row = {}
#         new_row['id'] = row['id']
#         options = ['', '', '']
#         ids = [0, 1, 2]
#         if row['label'] > 2:
#             print(row)
#         options[int(row['label'])] = str(row['label'] + 1) + '. ' + row['answer']
#         ids.remove(int(row['label']))
#         options[ids[0]] = str(ids[0] + 1) + '. ' + row['distractor1']
#         options[ids[1]] = str(ids[1] + 1) + '. ' + row['distractor2']
        
#         new_row['text'] = row['question'] + '\n' + '\n'.join(options)
#         new_row['label'] = str(row['label'] + 1)
#         final_data.append(new_row)

# df = pd.DataFrame(final_data)
# print(len(df))
# df.to_csv('brain_teaser.tsv', sep='\t', index=False)

# from refactor.metrics import analysis_debate_potential
# from refactor.dataset import ResultHandler

df = pd.read_csv('test.tsv', sep='\t', header=0)
df = df[['id', 'text', 'label']]
final_df = []
for _, row in df.iterrows():
    if len(row['label'].split(',')) > 1:
        labels = row['label'].split(',')
        for i in range(len(labels)):
            labels[i] = str(int(labels[i]) + 1)
        row['label'] = ','.join(labels)
        final_df.append(row)
final_df = pd.DataFrame(final_df).head(100)
final_df.to_csv('multi-label-emotion.tsv', sep='\t', index=False)
# r1 = ResultHandler(r'F:\NLP-lab\MultiAgentLLM\experiments\brain_teaser\full-experiment5\result\classification-result.tsv')
# r2 = ResultHandler(r'F:\NLP-lab\MultiAgentLLM\experiments\brain_teaser\full-experiment4\result\classification-result.tsv')
# r3 = ResultHandler(r'F:\NLP-lab\MultiAgentLLM\experiments\brain_teaser\full-experiment\result\classification-result.tsv')

# analysis_debate_potential(r1.predicts, r2.predicts, r3.predicts, r1.labels)