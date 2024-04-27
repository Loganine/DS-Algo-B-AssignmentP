# DS Algo Cheatsheet
## 一、可能用到的非课程知识点：

### 1.defaultdict可以用来创建键去重的字典。
引用方式:from collections import defaultdict
### 2.A是一个列表，A[~i]表示A的倒数第i项。
### 3.enumerate函数可以实现对一个数组中的元素匹配对应的指标。
### 4.permutations函数可以呈现数组中任取若干项排列组合
引用方式：from itertools import permutations
## 二、可能用到的本课程知识点：
### 1.类
要规定init等函数，很重要但是很难。
### 2.栈与队列
栈：一端进一端出；队列：一端进另一端出
```python
class Stack:
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items==[]
    def Push(self,item):
        self.items.append(item)
    def Pop(self):
        self.items.pop()
    def peek(self):
        return self.items[len(self.items)-1]
    def size(self):
        return len(self.items)
```
```python
class Queue:#前进后出
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items==[]
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self,item):
        self.items.pop()
    def size(self):
        return len(self.items)
```
### 3.树
掌握递归定义，节点，边，父子兄弟节点，层级，深度等概念(相关代码见下)
#### (1)哈夫曼树
```python
import heapq
class TreeNode:
    def __init__(self,weight,char=None):
        self.weight=weight
        self.char=char
        self.left=None
        self.right=None
    def __lt__(self,other):
        if self.weight==other.weight:
            return self.char<other.char
        return self.weight<other.weight
def build_huffman_tree(characters):
    heap=[]
    for char,weight in characters.items():
        heapq.heappush(heap,TreeNode(weight,char))
    while len(heap)>1:
        left=heapq.heappop(heap)
        right=heapq.heappop(heap)
        merged=TreeNode(left.weight+right.weight,min(left.char,right.char))
        merged.left=left
        merged.right=right
        heapq.heappush(heap,merged)
    return heap[0]
def encode_huffman_tree(root):
    codes={}
    def traverse(node,code):
        if node.left is None and node.right is None:
            codes[node.char]=code
        else:
            traverse(node.left,code+'0')
            traverse(node.right,code+'1')
    traverse(root,'')
    return codes
def huffman_encoding(codes,string):
    encoded=''
    for char in string:
        encoded+=codes[char]
    return encoded
def huffman_decoding(root,encoded_string):
    decoded=''
    node=root
    for bit in encoded_string:
        if bit=='0':
            node=node.left
        else:
            node=node.right
        if node.left is None and node.right is None:
            decoded+=node.char
            node=root
    return decoded
n=int(input())
characters={}
for i in range(n):
    char,weight=input().split()
    characters[char]=int(weight)
huffman_tree=build_huffman_tree(characters)
codes=encode_huffman_tree(huffman_tree)
while True:
    try:
        s=input()
        if s[0] in '01':
            print(huffman_decoding(huffman_tree,s))
        else:
            print(huffman_encoding(codes,s))
    except EOFError:
        break
```

#### (2)平衡二叉搜索树(AVL树)

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
class AVL:
    def __init__(self):
        self.root = None
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)
    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right)
        balance = self._get_balance(node)
        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    def preorder(self):
        return self._preorder(self.root)
    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
n = int(input().strip())
sequence = list(map(int, input().strip().split()))
avl = AVL()
for value in sequence:
    avl.insert(value)
print(' '.join(map(str, avl.preorder())))
```
### 4.排序
#### (1)冒泡排序
左右大小相反就互换位置，平均复杂度、最坏复杂度o(n^2)；
#### (2)选择排序
每次遍历把最小的放在未排好的第一项，平均复杂度、最坏复杂度o(n^2);
#### (3)插入排序
从前往后把每一项插入前面已经排好的序列的正确位置，平均复杂度、最坏复杂度o(n^2);
#### (4)希尔排序
分组插入排序，然后合并，平均复杂度o(n^4/3)，最坏复杂度o(n^3/2);
#### (5)快排
选取一个基准，比基准大的放右边，小的放左边，递归处理，平均复杂度o(nlogn),最坏复杂度o(n^2);
#### (6)归并排序
申请额外空间，双指针移动，小的先放，大的后放，分治策略，平均复杂度、最坏复杂度o(nlogn)
### 5.堆（二叉堆）
子节点永远小于父节点。
```python
class Binheap:
    def __init__(self):
        self.heaplist=[0]
        self.currentsize=0
    def percup(self,i):
        while i//2>0:
            if self.heaplist[i]<self.heaplist[i//2]:
                tmp=self.heaplist[i//2]
                self.heaplist[i//2]=self.heaplist[i]
                self.heaplist[i]=tmp
            i=i//2
    def insert(self,k):
        self.heaplist.append(k)
        self.currentsize+=1
        self.percup(self.currentsize)
    def percdown(self,i):
        while (i*2)<=self.currentsize:
            mc=self.minchild(i)
            if self.heaplist[i]>self.heaplist[mc]:
                tmp=self.heaplist[i]
                self.heaplist[i]=self.heaplist[mc]
                self.heaplist[mc]=tmp
            i=mc
    def minchild(self,i):
        if i*2+1>self.currentsize:
            return i*2
        else:
            if self.heaplist[i*2]<self.heaplist[i*2+1]:
                return i*2
            else:
                return i*2+1
    def delmin(self):
        retval=self.heaplist[1]
        self.heaplist[1]=self.heaplist[self.currentsize]
        self.currentsize-=1
        self.heaplist.pop()
        self.percdown(1)
        return retval
    def buildheap(self,alist):
        i=len(alist)//2
        self.currentsize=len(alist)
        self.heaplist=[0]+alist[:]
        while i>0:
            self.percdown(i)
            i-=1
```
### 6.图
```python
class Vertex:
    def __init__(self,key):
        self.id=key
        self.connectedto={}
    def addneighbor(self,nbr,weight=0):
        self.connectedto[nbr]=weight
    def __str__(self):
        return str(self.id)+' connectedto: '+str([x.id for x in self.connectedto])
    def getconnections(self):
        return self.connectedto.keys()
    def getid(self):
        return self.id
    def getweight(self,nbr):
        return self.connectedto[nbr]
class Graph:
    def __init__(self):
        self.vertlist={}
        self.numVertices=0
    def addVertex(self,key):
        self.numVertices=self.numVertices+1
        newVertex=Vertex(key)
        self.vertlist[key]=newVertex
        return newVertex
    def getVertex(self,n):
        if n in self.vertlist:
            return self.vertlist[n]
        else:
            return None
    def __contains__(self,n):
        return n in self.vertlist
    def addedge(self,f,t,weight=0):
        if f not in self.vertlist:
            nv=self.addVertex(f)
        if t not in self.vertlist:
            nv=self.addVertex(t)
        self.vertlist[f].addneighbor(self.vertlist[t],weight)
    def gervertices(self):
        return self.vertlist.keys()
    def __iter__(self):
        return iter(self.vertlist.values())
```
## 三、可能用到的源代码：
### 1.快排

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)

def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]#选定基准
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i

arr = [22, 11, 88, 66, 55, 77, 33, 44]#这里是举个例子
quicksort(arr, 0, len(arr) - 1)
print(arr)


```



### 2.归并排序

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2
		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves



​	mergeSort(L) # Sorting the first half
​	mergeSort(R) # Sorting the second half

​	i = j = k = 0

	# Copy data to temp arrays L[] and R[]

​	while i < len(L) and j < len(R):
​		if L[i] <= R[j]:
​			arr[k] = L[i]
​			i += 1
​		else:
​			arr[k] = R[j]
​			j += 1
​		k += 1

	# Checking if any element was left

​	while i < len(L):
​		arr[k] = L[i]
​		i += 1
​		k += 1

​	while j < len(R):
​		arr[k] = R[j]
​		j += 1
​		k += 1

注:如果需要同时知道数组的逆序数，可采用如下代码

def mergesort(a):
    if len(a)<=1:
        return a,0
    mid=len(a)//2
    left,nixuleft=mergesort(a[:mid])
    right,nixuright=mergesort(a[mid:])
    merged,nixumerge=merge(left,right)
    return merged,nixuleft+nixuright+nixumerge
def merge(left,right):
    merged=[]
    ans=0
    i=j=0
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            merged.append(left[i])
            i+=1
        else:
            merged.append(right[j])
            j+=1
            ans+=len(left)-i#考虑的是两个之间的逆序数
    merged+=left[i:]#如果还有剩余就放回新列表里面
    merged+=right[j:]
    return merged,ans #返回值为排好序的列表与逆序数
```



### 3.只求逆序数（逆天版本）这里用到的是bisect库。
```python
from bisect import *
n=int(input())
ans=0
lis=[int(i) for i in input().split()]
new=[]
for i in lis:
    pos=bisect_left(new,i)
    ans+=pos
    insort_left(new,i)
print(int(n*(n-1)/2-ans))
```



### 4.八皇后问题（可扩展为n皇后问题）            
```python
def get_queen_answers(n):
    answers=[]
    queens=[-1]*n
    def backtrack(hang):
        if hang==n:
            answers.append(queens.copy())
        else:
            for lie in range(n):
                if is_valid(hang,lie):
                    queens[hang]=lie#put the queen(hang) to the lieth col
                    backtrack(hang+1)#back to the next row
                    queens[hang]=-1
    def is_valid(hang,lie):
        for h in range(hang):
            if queens[h]==lie or abs(lie-queens[h])==abs(hang-h):
                return False
        return True
    backtrack(0)
    return answers
def get_needed_queens(b):
    ans=get_queen_answers(8)
    if b>len(ans):
        return None
    needed_queens=''.join(str(lie+1)for lie in ans[b-1])
    return needed_queensy
```


### 5.约瑟夫问题（存在变式）
```python
def Josephus(names,m):
    queue=[]
    for i in names:
        queue.append(i)
    while len(queue)>1:
        for i in range(m):
            queue.append(queue.pop(0))
        queue.pop(0)
    return queue[0]
(从第一个人开始数)
```

  

```python
  n,p,m=map(int,input().split())
    if n==m==p==0:
        break
    num=[i for i in range(1,n+1)]
    for i in range(p-1):
        temp=num.pop(0)
        num.append(temp)
    index=0
    ans=[]
    while len(num)>1:
        temp=num.pop(0)
        index+=1
        if index==m:
            index=0
            ans.append(temp)
            continue
        num.append(temp)
    ans.append(num[0])
```
### 6.递归



#### (1)斐波那契数列
```python
from functools import lru_cache
@lru_cache(maxsize=2000)
def fib(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)
```


#### (2)汉诺塔问题
```python
def moveone(num:int,start:str,end:str):
    print('{}:{}->{}'.format(num,start,end))
def move(num:int,start:str,middle:str,end:str):
    if num==1:
        moveone(1,start,end)
    else:
        move(num-1,start,end,middle) #从出发点经过终点将n-1个木块放在中间点
        moveone(num,start,end)
        move(num-1,middle,start,end) #从中间点经过出发点将n-1个木块放在终点
n,a,b,c=input().split()
move(int(n),a,b,c)
```


#### (3)谢尔宾斯基分形三角形(the Sierbinski Fractal)
```python
def f(n):
    if n==1:
        return [' /\\ ','/__\\']
    t=f(n-1)
    x=2**(n-1)
    res=[' '*x + u + ' '*x for u in t]
    res.extend(u+u for u in t)
    return res
```


#### (4)dfs 括号生成 (同时可以看作合法出栈序列的进栈出栈顺序)
```python
ans=[]
def dfs(n:int,groups:str,left:int,right:int):
    global ans
    if len(groups)==2*n:
        ans.append(groups)
    for i in '()':
        if i=='(':
            if left<n:
                dfs(n,groups+i,left+1,right)
        else:
            if right<left and right<n:
                dfs(n,groups+i,left,right+1)
dfs(int(input()),'',0,0)
```


#### (5)树 
##### [1]二叉树的深度和叶子数目
```python
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

def tree_height(node):
    if node is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(node.left), tree_height(node.right)) + 1

def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

n = int(input())  # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
        has_parent[left_index] = True
    if right_index != -1:
        print(right_index)
        nodes[i].right = nodes[right_index]
        has_parent[right_index] = True#寻找根节点，也就是没有父节点的节点

root_index = has_parent.index(False)
root = nodes[root_index]

    # 计算高度和叶子节点数

height = tree_height(root)
leaves = count_leaves(root)
```


##### [2]后序表达式改队列表达式
```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
def buildtree(s):
    stack=[]
    

for char in s:
    node=TreeNode(char)
    if 'A'<=char<='Z':
        node.right=stack.pop()
        node.left=stack.pop()
    stack.append(node)
return stack[0]

def level_order_traversal(root):
    queue=[root]
    traversal=[]
    while queue:
        node=queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal
n=int(input().strip())
for i in range(n):
    s=input().strip()
    root=buildtree(s)
    print(''.join(level_order_traversal(root)[::-1]))

#### 
```

##### [3]前中序建树转后序
```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
def buildtree(pre,mid):
    if not pre or not mid:
        return None
    root_value=pre[0]
    root=TreeNode(root_value)
    root_midorder=mid.index(root_value)
    root.left=buildtree(pre[1:1+root_midorder],mid[:root_midorder])
    root.right=buildtree(pre[root_midorder+1:],mid[1+root_midorder:])
    return root
def postorder(root):
    if root is None:
        return ''
    return postorder(root.left)+postorder(root.right)+root.value
```

##### [4]万能树

```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
    def traversal(self,mode):
        result=[]
        if mode=='preorder':
            result.append(self.value)
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode=='postorder':
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            result.append(self.value)
            return result
        elif mode=='inorder':
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            result.append(self.value)
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode=='levelorder':
            queue=[self]
            while queue:
                node=queue.pop(0)
                result.append(node.value)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            return result
def qzbuildtree(pre,mid):
    if not pre or not mid:
        return None
    root_value=pre[0]
    root=TreeNode(root_value)
    root_midorder=mid.index(root_value)
    root.left=qzbuildtree(pre[1:1+root_midorder],mid[:root_midorder])
    root.right=qzbuildtree(pre[root_midorder+1:],mid[1+root_midorder:])
    return root
def zhbuildtree(mid,post):
    if not post or not mid:
        return None
    root_value=post[len(post)-1]
    root=TreeNode(root_value)
    root_midorder=mid.index(root_value)
    root.right=zhbuildtree(mid[root_midorder+1:],post[root_midorder:len(post)-1])
    root.left=zhbuildtree(mid[:root_midorder],post[:root_midorder])
    return root
```
#### (6)拉链
```python
def dfs(x,y):
    if x==-1 and y==-1:
        return True
    if x>=0 and a[x]==c[x+y+1]:
        if dfs(x-1,y):
            return True
    if y>=0 and b[y]==c[x+y+1]:
        if dfs(x,y-1):
            return True
    return False
for i in range(int(input())):
    a,b,c=input().split()
    if dfs(len(a)-1,len(b)-1):
        print(f'Data set {i+1}: yes')
    else:
        print(f'Data set {i+1}: no')
```
