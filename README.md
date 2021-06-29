## How to run
run in src directory
<pre><code>python3 AnomalyGAN.py</code></pre>

#### Dataset Techniques
Adjacency matrix, feature matrix and label should be placed one after the other with format:

1. Feature matrix
<pre>    
Size X Y                      # X->no of vertices, Y->feature vector size of one vertex
a b c d                       # [1]->a, b, c, d
-1 -1
</pre>

2. Adjacency matrix
<pre>    
Size X                        # X->no of vertices
a b                           # edge from a to b
-1 -1
</pre>    

3. Label Matrix
<pre>    
0/1      # 0: non-anomalous/1:anomalous
</pre>