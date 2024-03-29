# 一、可能用到的非课程知识点：

## 1.defaultdict可以用来创建键去重的字典。
引用方式:from collections import defaultdict
## 2.A是一个列表，A[~i]表示A的倒数第i项。
## 3.enumerate函数可以实现对一个数组中的元素匹配对应的指标。
## 4.permutations函数可以呈现数组中任取若干项排列组合
引用方式：from itertools import permutations
# 二、可能用到的本课程知识点：
## 1.类
要规定init等函数，很重要但是很难。
## 2.栈与队列
栈：一端进一端出；队列：一端进另一端出
```python
class Stack:
    def __init__(self):
        self.items=[]
    def is_empty(self):
        return self.items=[]
    def Push(self,item):
        self.items.append(item)
    def Pop(self):
        self.items.pop()
    def peek(self):
        return self.items[len(self.items)-1]
    def size(self):
        return len(self.items)
## 3.树
掌握递归定义，节点，边，父子兄弟节点，层级，深度等概念
## 4.排序
### (1)冒泡排序：左右大小相反就互换位置，平均复杂度、最坏复杂度o(n^2)；
### (2)选择排序：每次遍历把最小的放在未排好的第一项，平均复杂度、最坏复杂度o(n^2);
### (3)插入排序：从前往后把每一项插入前面已经排好的序列的正确位置，平均复杂度、最坏复杂度o(n^2);
### (4)希尔排序：分组插入排序，然后合并，平均复杂度o(n^4/3)，最坏复杂度o(n^3/2);
### (5)快排：选取一个基准，比基准大的放右边，小的放左边，递归处理，平均复杂度o(nlogn),最坏复杂度o(n^2);
### (6)归并排序:申请额外空间，双指针移动，小的先放，大的后放，分治策略，平均复杂度、最坏复杂度o(nlogn)
# 三、可能用到的源代码：
## 1.快排

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



## 2.归并排序

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



## 3.只求逆序数（逆天版本）这里用到的是bisect库。
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



## 4.八皇后问题（可扩展为n皇后问题）            
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


## 5.约瑟夫问题（存在变式）
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

## 6.递归
```



### (1)斐波那契数列
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


### (2)汉诺塔问题
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


### (3)谢尔宾斯基分形三角形(the Sierbinski Fractal)
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


### (4)dfs 括号生成 (同时可以看作合法出栈序列的进栈出栈顺序)
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


### (5)树 
#### [1]二叉树的深度和叶子数目
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


#### [2]后序表达式改队列表达式
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

#### [3]前中序建树转后序
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

#### [4]万能树

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

