import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\RUSHU\Downloads\sports_articles_dataset.csv")

data.columns = data.columns.str.strip().str.lower()


print("Columns in CSV:", data.columns.tolist())

text_column = 'article' 
if text_column not in data.columns:
 
    text_column = [col for col in data.columns if 'text' in col or 'article' in col][0]

articles = data[text_column].dropna().tolist()

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(articles)

true_k = 3
model = KMeans(n_clusters=true_k, random_state=42, n_init=10)
model.fit(X)

terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

print("\nTop terms per cluster:\n")
for i in range(true_k):
    print(f"Cluster {i}:")
    for ind in order_centroids[i, :5]:
        print(f"  {terms[ind]}")
    print()

svd = TruncatedSVD(n_components=2, random_state=42)
reduced_data = svd.fit_transform(X)


plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1],
            c=model.labels_, cmap='rainbow', s=80)
plt.title("Sports Article Clustering using K-Means")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()


new_articles = [
    "India defeated Pakistan in the T20 cricket match.",
    "Nadal wins French Open tennis tournament again.",
    "Chelsea signed a new striker for the Premier League."
]

Y = vectorizer.transform(new_articles)
predicted = model.predict(Y)

print("\nPredicted clusters for new articles:\n")
for text, cluster in zip(new_articles, predicted):
    print(f"'{text}' => Cluster {cluster}")
