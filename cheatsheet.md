# DS Algo Cheatsheet
## 一、可能用到的非课程知识点：

### 1.defaultdict
可以用来创建具有默认值的字典。例如int默认值为0，list默认值为[]等。
引用方式:from collections import defaultdict
### 2.A[~i]
A是一个列表，A[~i]表示A的倒数第i项。
### 3.enumerate函数
可以实现对一个数组中的元素匹配对应的指标。
### 4.permutations函数
可以呈现数组中任取若干项排列组合
引用方式：from itertools import permutations
## 二、可能用到的本课程知识点：
### 1.类
要规定init等方法，很重要但是很难。
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



### 4.约瑟夫问题（存在变式）
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
### 5.递归



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
##### [5] 括号嵌套树
```python
class Treenode:
    def __init__(self,value):
        self.value=value
        self.children=[]
def parse_tree(s):
    stack=[]
    node=None
    for char in s:
        if char.isalpha():
            node=Treenode(char)
            if stack:#栈如果非空，就将节点作为子节点加入栈顶结点的子节点列表中
                stack[-1].children.append(node)
        elif char=='(':
            if node:
                stack.append(node)
                node=None
        elif char==')':
            if stack:
                node=stack.pop()
    return node #根节点
def preorder(node):
    output=[node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)
def postorder(node):
    output=[]
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)
s=input().strip()
s=''.join(s.split())
print(preorder(parse_tree(s)))
print(postorder(parse_tree(s)))
```
#### (6)八皇后问题（n皇后问题）
八皇后问题（可扩展为n皇后问题）            
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
    return needed_queens
```



#### (7)拉链
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
#### (8)两座孤岛最短距离
```python
from collections import deque
def dfs(x,y,grid,n,queue,directions):
    grid[x][y]=2
    queue.append((x,y))
    for dx,dy in directions:
        nx,ny=x+dx,y+dy
        if 0<=nx<n and 0<=ny<n and grid[nx][ny]==1:
            dfs(nx,ny,grid,n,queue,directions)
def bfs(grid,n,queue,directions):
    dis=0
    while queue:
        for _ in range(len(queue)):
            x,y=queue.popleft()
            for dx,dy in directions:
                nx,ny=x+dx,y+dy
                if 0<=nx<n and 0<=ny<n:
                    if grid[nx][ny]==1:
                        return dis
                    elif grid[nx][ny]==0:
                        grid[nx][ny]=2
                        queue.append((nx,ny))
        dis+=1
    return dis
n=int(input())
grid=[list(map(int,input())) for _ in range(n)]
directions=[(1,0),(-1,0),(0,1),(0,-1)]
queue=deque()
def find(grid):
    for i in range(n):
        for j in range(n):
            if grid[i][j]==1:
                dfs(i,j,grid,n,queue,directions)
                return bfs(grid,n,queue,directions)
print(find(grid))
```
#### (9) 算鹰
```python
def dfs(x,y):
    if x<0 or y<0 or x>9 or y>9 or board[x][y]!='.':
        return
    directions=[(0,1),(0,-1),(-1,0),(1,0)]
    if board[x][y]=='.':
        board[x][y]='-'
        for direction in directions:
            nx,ny=x+direction[0],y+direction[1]
            dfs(nx,ny)
board=[list(input()) for _ in range(10)]
cnt=0
for i in range(10):
    for j in range(10):
        if board[i][j]=='.':
            dfs(i,j)
            cnt+=1
print(cnt)
```

### 6.图（包含bfs）
#### (1)词梯
```python
from collections import defaultdict
dic=defaultdict(list)
n,lis=int(input()),[]
for i in range(n):
    lis.append(input())
for word in lis:
    for i in range(len(word)):
        bucket=word[:i]+'_'+word[i+1:]
        dic[bucket].append(word)
def bfs(start,end,dic):
    queue=[(start,[start])]
    visited=[start]
    while queue:
        currentword,currentpath=queue.pop(0)
        if currentword==end:
            return ' '.join(currentpath)
        for i in range(len(currentword)):
            bucket=currentword[:i]+'_'+currentword[i+1:]
            for nbr in dic[bucket]:
                if nbr not in visited:
                    visited.append(nbr)
                    newpath=currentpath+[nbr]
                    queue.append((nbr,newpath))
    return 'NO'
start,end=map(str,input().split())    
print(bfs(start,end,dic))
```
#### (2)骑士周游
这里用到了Warnsdorff算法，即优先探索难以达到的位置，即下列代码中的ordered_by_avail函数。
```python
class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0

    def __lt__(self,o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
            self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [
        (-1, -2),  # left-down-down
        (-1, 2),  # left-up-up
        (-2, -1),  # left-left-down
        (-2, 1),  # left-left-up
        (1, -2),  # right-down-down
        (1, 2),  # right-up-up
        (2, -1),  # right-right-down
        (2, 1),  # right-right-up
    ]
    for r_off, c_off in move_offsets:
        if (
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

def pos_to_node_id(x, y, bdSize):
    return x * bdSize + y

def legal_coord(row, col, board_size):
    return 0 <= row < board_size and 0 <= col < board_size



def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)
    if n < limit:
        neighbors = ordered_by_avail(u)
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):
                return True
        else:
            path.pop()
            u.color = "white"
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0

    def dfs(self):
        for vertex in self:
            vertex.color = "white"
            vertex.previous = -1
        for vertex in self:
            if vertex.color == "white":
                self.dfs_visit(vertex)

    def dfs_visit(self, start_vertex):
        start_vertex.color = "gray"
        self.time = self.time + 1
        start_vertex.discovery_time = self.time
        for next_vertex in start_vertex.get_neighbors():
            if next_vertex.color == "white":
                next_vertex.previous = start_vertex
                self.dfs_visit(next_vertex)
        start_vertex.color = "black"
        self.time = self.time + 1
        start_vertex.closing_time = self.time


def main():
    def NodeToPos(id):
       return ((id//8, id%8))

    bdSize = int(input())  # 棋盘大小
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdSize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdSize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdSize * bdSize-1)
    if done:
        print("success")
    else:
        print("fail")

    #exit(0)

    # 打印路径
    cnt = 0
    for vertex in tour_path:
        cnt += 1
        if cnt % bdSize == 0:
            print()
        else:
            print(vertex.key, end=" ")
            #print(NodeToPos(vertex.key), end=" ")   # 打印坐标

if __name__ == '__main__':
    main()
```
#### (3)最小生成树Kruskal算法
运用并查集避免成环。
```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1


def kruskal(graph):
    num_vertices = len(graph)
    edges = []

    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    # 按照权重排序
    edges.sort(key=lambda x: x[2])

    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)

    # 构建最小生成树的边集
    minimum_spanning_tree = []

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))

    return minimum_spanning_tree
```
### 7.栈的应用
#### （1）调度场算法
用于前中后序表达式的互相转换。
```python
#中序转后序
def zhongzhuanhou(string):
    precedence={"+":1,'-':1,'*':2,'/':2}
    number=''
    stack=[]
    answer=[]
    for char in string:
        if char.isnumeric() or char=='.':
            number+=char
        else:
            if number:
                num=float(number)
                answer.append(int(num) if num.is_integer() else num)
                number=''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char]<=precedence[stack[-1]]:
                    answer.append(stack.pop())
                stack.append(char)
            elif char=='(':
                stack.append(char)
            elif char==')':
                while stack and stack[-1]!="(":
                    answer.append(stack.pop())
                stack.pop()
    if number:
        num=float(number)
        answer.append(int(num) if num.is_integer() else num)
    while stack:
        answer.append(stack.pop())
    return ' '.join(str(x) for x in answer)
n=int(input())
for i in range(n):
    s=input()
    print(zhongzhuanhou(s))
```
```python
#前缀表达式（波兰表达式）求值
def Operator(num1,num2,operator):
    if operator == '+':
        result = float(num1) + float(num2)
    elif operator == '-':
        result = float(num1) - float(num2)
    elif operator == '*':
        result = float(num1) * float(num2)
    else:
        if num2 == '0.0':
            return 'ERROR'       
        result = float(num1) / float(num2)
    return str(result)
def DealExpression(strs):
    stack = []
    operator = ['+','-','*','/']
    length = len(strs)
    if length == 1:
        result = float(strs[0])
    elif length == 2:
        result = Operator('0',strs[1],strs[0])
    else:
        for i in range(length-1,-1,-1):
            if strs[i] not in operator:
                stack.append(strs[i])
            if strs[i] in operator:
                num1 = stack.pop()
                num2 = stack.pop()
                result = Operator(num1,num2,strs[i])
                if result == 'ERROR':
                    return result
                stack.append(result)
        result = stack[0]

    return result

strs = input().split(" ")
result = DealExpression(strs)
if result == 'ERROR':
    print('ERROR')
else:print("%.6f"%float(result))
```
```python
#上述题目的简单解法（代码简单）
def cal(s):
    t=s.pop(0)
    if t in '+-*/':
        return str(eval(cal(s)+t+cal(s)))
    else: return t
s=input().split()
print(f'{float(cal(s)):.6f}')
```
### 8.图
#### （1）Dijkstra算法（以兔子与樱花为例）
```python
import heapq
def dijkstra(adjacency,start):
    distances={vertex:float('infinity') for vertex in adjacency}
    prev={vertex:None for vertex in adjacency}
    distances[start]=0
    pq=[(0,start)]
    while pq:
        curdis,curv=heapq.heappop(pq)
        if curdis>distances[curv]:
            continue
        for nbr,weight in adjacency[curv].items():
            dis=curdis+weight
            if dis<distances[nbr]:
                distances[nbr]=dis
                prev[nbr]=curv
                heapq.heappush(pq,(dis,nbr))
    return distances,prev
def shortestpath(adjacency,start,end):
    distances,prev=dijkstra(adjacency,start)
    path=[]
    current=end
    while prev[current] is not None:
        path.insert(0,current)
        current=prev[current]
    path.insert(0,start)
    return path,distances[end]
p=int(input())
places={input().strip() for _ in range(p)}
q=int(input())
graph={place:{} for place in places}
for _ in range(q):
    start,end,dis=input().split()
    dis=int(dis)
    graph[start][end]=dis
    graph[end][start]=dis
r=int(input())
ops=[input().split() for _ in range(r)]
for start,end in ops:
    if start==end:
        print(start)
        continue
    path,max_dis=shortestpath(graph,start,end)
    cout=''
    for i in range(len(path)-1):
        cout+=f'{path[i]}->({graph[path[i]][path[i+1]]})->'
    cout+=f'{end}'
    print(cout)
```
#### （2）kruskal算法(最小生成树权重和)
```python
X,R={},{}
def disjointset(char):
    X[char]=char
    R[char]=0
def find(x):
    if X[x]!=x:
        X[x]=find(X[x])
    return X[x]
def union(start,end):
    r1=find(start)
    r2=find(end)
    if r1!=r2:
        if R[r1]>R[r2]:
            X[r2]=r1
        else:
            X[r1]=r2
            if R[r1]==R[r2]:
                R[r2]+=1
def kruskal(dic):
    edges=[]
    for i in dic.keys():
        disjointset(i)
        for j in dic[i].keys():
            if ord(i)<ord(j):
                edges.append((dic[i][j],i,j))
    edges=sorted(edges,key=lambda x:x[0])
    ans=0
    for edge in edges:
        weight,start,end=edge
        if find(start)!=find(end):
            union(start,end)
            ans+=weight
    return ans
```
#### （3）prim算法（最小生成树权重和）
```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()
```
