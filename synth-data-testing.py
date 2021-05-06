from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Load both synth and real datasets
# 4096-comp sets
synth_4096 = np.load('./synthetic_data/20210506-1207154096-comp-synth-square-shape.npy')
synth_4096 = np.reshape(synth_4096, (synth_4096.shape[0], 4096))
real_4096 = np.load('./pca_data/20210506-1207174096-comp-pivot-table-square-shape.npy')
real_4096 = np.reshape(real_4096, (real_4096.shape[0], 4096))
# reconstructed sets
synth_reconstructed = np.load('./synthetic_data/20210506-120740reconstructed-synth.npy')
real_reconstructed = np.load('./pca_data/20210506-120736reconstructed-pivot-table-flattened.npy')
# real data in npy format
real = np.load('./real_data/20210506-110621real-data-flattened.npy')

# Plot 4096 synth and 4096 real on a scatter plot
# plt.plot(synth_4096[:100, 0], synth_4096[:100, 1], 'ro')
# plt.plot(real_4096[:100, 0], real_4096[:100, 1], 'bo')
# plt.show()
fig = plt.figure(figsize=(4,4))
fig.suptitle('Principle Components of Real and Synthetic Data Plot', fontsize=16)
plot_data = [(synth_4096[:100, i], synth_4096[:100, j], real_4096[:100, i], real_4096[:100, j]) if i != j else ([],[],[],[]) for j in range(4) for i in range(4)]
for i in range(16):
    ax = plt.subplot(4,4,i+1, xlabel='PC {}'.format(int(i/4)), ylabel='PC {}'.format(i%4))
    s_i, s_j, r_i, r_j = plot_data[i]
    if(int(i/4) != i%4):
        # a = plt.plot(s_i, s_j, 'ro', markersize=5)
        # b = plt.plot(r_i, r_j, 'bo', markersize=5)
        ax.plot(s_i, s_j, 'ro', markersize=5, label='Synthetic')
        ax.plot(r_i, r_j, 'bo', markersize=5, label='Real')
        ax.legend(loc='upper right')
        #fig.legend((a,b), ('Synthetic', 'Real'), 'upper right')
#plt.tight_layout(pad=0.1, w_pad=1, h_pad=1)
#plt.legend()
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
plt.show()



# Construct three full datasets w/ synth and real data and (0, 1) labels for synth and real, respectively
synth_4096 = np.append(synth_4096, np.zeros((synth_4096.shape[0], 1)), axis=1)
real_4096 = np.append(real_4096, np.ones((real_4096.shape[0], 1)), axis=1)
full_4096 = np.concatenate((synth_4096, real_4096))

# fig = px.scatter_matrix(
#     full_4096[:, :-1],
#     labels=range(4),
#     dimensions=range(4),
#     color = full_4096[:, -1]
# )

# fig.update_traces(diagonal_visible=False)
# fig.show()

synth_reconstructed = np.append(synth_reconstructed, np.zeros((synth_reconstructed.shape[0], 1)), axis=1)
real_reconstructed = np.append(real_reconstructed, np.ones((real_reconstructed.shape[0], 1)), axis=1)
full_reconstructed  = np.concatenate((synth_reconstructed, real_reconstructed))

real = np.append(real, np.ones((real.shape[0], 1)), axis=1)
full_real_and_reconstructed = np.concatenate((synth_reconstructed, real))
# Build three train-test splits
train_sets = []
test_sets = []

for n, d in [('pca_4096', full_4096),
            ('pca_reconstructed', full_reconstructed),
            ('real_and_pca_reconstructed', full_real_and_reconstructed)]:
    train, test = train_test_split(d, test_size=0.2)
    train_sets.append((n, train))
    test_sets.append((n, test))

# Build pipelines out of the various classifiers and plug them into GridSearchCV's
pipelines = []
models = [GaussianNB()]

# For dataset, fit the models, print test results
for (train_n, train), (test_n, test) in zip(train_sets, test_sets):
    train_X = train[:, :-1]
    train_y = train[:, -1]
    test_X = test[:, :-1]
    test_y = test[:, -1]
    for model in models:
        model.fit(train_X, train_y)
        print('Mean accuracy of GNB on dataset {} = {}'.format(train_n, model.score(test_X, test_y)))
