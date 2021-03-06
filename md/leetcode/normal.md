## 力扣normal总结

#### [1573. 分割字符串的方案数](https://leetcode-cn.com/problems/number-of-ways-to-split-a-string/)

题目：

给你一个二进制串 s  （一个只包含 0 和 1 的字符串），我们可以将 s 分割成 3 个 非空 字符串 s1, s2, s3 （s1 + s2 + s3 = s）。

请你返回分割 s 的方案数，满足 s1，s2 和 s3 中字符 '1' 的数目相同。

由于答案可能很大，请将它对 10^9 + 7 取余后返回。



解答：

找出分界点，然后统计第一个均分1的结尾位置，第二个均分1的起始位置，长度计为a；第二个均分1结尾的位置，第三个均分1的起始位置，长度计为b；然后计算a * b即可。

``` Java
class Solution {
    public int numWays(String s) {
        char[] chars = s.toCharArray();
        int count = 0;
        for (char c : chars) {
            if (c == '1') ++ count;
        }
        if (count == 0) {
            long a = chars.length - 1;
            long b = chars.length - 2;
            return (int) ((a * b) / 2 % 1000000007);
        } else if (count % 3 != 0) {
            return 0;
        } else {
            int k = count / 3;
            long a = 0, b = 0;
            int tmp = 0;
            int i;
            for (i = 0; i < chars.length; ++ i) {
                if (chars[i] == '1') ++ tmp;
                if (tmp == k) {
                    break;
                }
            }
            for (i = i + 1;i < chars.length; ++ i) {
                if (chars[i] != '1') ++ a;
                else {
                    break;
                }
            }
            tmp = 0;
            for (; i < chars.length; ++ i) {
                if (chars[i] == '1') ++ tmp;
                if (tmp == k) {
                    break;
                }
            }
            for (i = i + 1; i < chars.length; ++ i) {
                if (chars[i] != '1') ++ b;
                else {
                    break;
                }
            }
            return (int) (((a + 1) * (b + 1)) % 1000000007);
        }
    }
}
```

#### [1574. 删除最短的子数组使剩余数组有序](https://leetcode-cn.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/)

题目：

给你一个整数数组 arr ，请你删除一个子数组（可以为空），使得 arr 中剩下的元素是 非递减 的。

一个子数组指的是原数组中连续的一个子序列。

请你返回满足题目要求的最短子数组的长度。



解答：

我们分组讨论即可，首先是需要删除的子数组在最左边，其次是在最右边，然后是中间，但是我们需要注意一下，中间找到的子数组需要我们再判断一次，即两头扩展，找到最短子数组。

``` Java
class Solution {
    public int findLengthOfShortestSubarray(int[] arr) {
        if (arr.length == 1) {
            return 0;
        }
        int a = left(arr);
        int b = right(arr);
        int c = middle(arr);
        int tmp = Math.min(a, b);
        return Math.min(tmp, c);
    }

    private int left(int[] arr) {
        int back = arr.length - 1;
        for (; back > 0; -- back) {
            if (arr[back - 1] > arr[back]) {
                break;
            }
        }
        return back;
    }

    private int right(int[] arr) {
        int front = 0;
        for (; front < arr.length - 1; ++ front) {
            if (arr[front] > arr[front + 1]) {
                break;
            }
        }
        return arr.length - front - 1;
    }

    private int middle(int[] arr) {
        int len = arr.length;
        int front = len - 1, back = 0;
        for (int i = 0; i < len - 1; ++ i) {
            if (arr[i] > arr[i + 1]) {
                // 子数组前部分最后一个位置
                front = i;
                break;
            }
        }
        // 说明全部是有序的
        if (front == len - 1) {
            return 0;
        }
        for (int i = len - 1; i > 0; -- i) {
            if (arr[i - 1] > arr[i]) {
                back = i;
                break;
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int i = back; i < len; ++ i) {
            for (int j = front; j >= 0; -- j) {
                if (arr[j] <= arr[i]) {
                    System.out.println(i + " : " + j);
                    ans = Math.min(ans, i - j - 1);
                    break;
                }
            }
        }
        return ans;
    }
}
```

#### [684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

题目：

树可以看成是一个连通且 无环 的 无向 图。

给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。

请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。



**示例 1：**

![img](https://pic.leetcode-cn.com/1626676174-hOEVUL-image.png)



解答：

使用并查集，依据输入顺序进行合并，合并之前检查如果这两个节点已经处于一个集合中了，那这条连接就是一个冗余连接，我们记录一下，最后输出最后一个出现的冗余连接即可。

``` Java
class Solution {
    
    // 如果parent[index] == index，则index就是自己的boss。
    private int[] parent = new int[1010];

    int findParent(int index) {
        int tmpIndex = index;
        while (index != parent[index]) {
            index = parent[index];
        }
        parent[tmpIndex] = index;
        return index;
    }

    void merge(int from, int to) {
        parent[findParent(from)] = findParent(to);
    }

    void init() {
        int len = parent.length;
        for (int i = 0; i < len; ++ i) {
            parent[i] = i;
        }
    }

    boolean isInSet(int a, int b) {
        return findParent(a) == findParent(b);
    }

    public int[] findRedundantConnection(int[][] edges) {
        int m = edges.length;
        int[] ans = new int[m * 2];
        int index = 0;
        init();
        for (int i = 0; i < m; ++ i) {
            int u = edges[i][0];
            int v = edges[i][1];
            if (isInSet(u, v)) {
                ans[index++] = u;
                ans[index++] = v;
            }
            merge(v, u);
        }
        if (index == 0) {
            return edges[m - 1];
        } else {
            int[] rcd = new int[2];
            rcd[0] = ans[index - 2];
            rcd[1] = ans[index - 1];
            return rcd;
        }
    }
}
```

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

题目：

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。



解答：

通过并查集记录一下，每个岛屿所属于的集合，然后看有多少个集合即可，在这里我们我们需要记得取找寻邻接岛屿，然后加入集合，最后统计集合数量。

``` Java
import java.util.HashSet;
import java.util.Set;

class Solution {
    private int m;
    private int n;

    private int[] parent;

    private static final int[] X = {-1, 0, 1, 0};
    private static final int[] Y = {0, -1, 0, 1};

    private int findParent(int index) {
        int tmpIndex = index;
        while (index != parent[index])
            index = parent[index];
        parent[tmpIndex] = index;
        return index;
    }

    private void merge(int from, int to) {
        parent[findParent(from)] = findParent(to);
    }

    private void init() {
        int len = parent.length;
        for (int i = 0; i < len; ++ i) {
            parent[i] = i;
        }
    }

    public int numIslands(char[][] grid) {
        m = grid.length;
        n = grid[0].length;
        parent = new int[m * n + 1];
        init();
        for (int i = 0; i < m; ++ i) {
            for (int j = 0; j < n; ++ j) {
                for (int k = 0; k < 4; ++ k) {
                    int x = i + X[k];
                    int y = j + Y[k];
                    if (grid[i][j] == '1' && isInBound(x, y) && grid[x][y] == '1') {
                        // System.out.println(i + " : " + j + "  " + x + " : " + y);
                        // System.out.println((i * n + j) + " : " + (x * n + y));
                        merge(i * n + j, x * n + y);
                    }
                }
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < m; ++ i) {
            for (int j = 0; j < n; ++ j) {
                if (grid[i][j] == '1') {
                    set.add(findParent(i * n + j));
                }
            }
        }
        return set.size();
    }

    private boolean isInBound(int x, int y) {
        return (x >= 0) && (y >= 0) && (x < m) && (y < n);
    }
}
```

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

题目：

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `**k**` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。



解答：

我们对数组降序排序，然后找到第k个元素即可，但是还有更加快的方法，就是HashMap，我们记录一下每个元素出现的个数，同时记录最大值，然后从最大值开始递减，使用k - 个数，直到k最后的值落在了这个value对应的个数中，为了优化，我们可以引入优先队列快速找到下一个降序元素。

``` Java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        int max = Integer.MIN_VALUE;
        for (int j : nums) {
            max = Math.max(max, j);
            hashMap.merge(j, 1, Integer::sum);
        }
        while (true) {
            Integer num = hashMap.get(max);
            if (num == null) {
                -- max;
                continue;
            }
            if (k <= num) {
                return max;
            } else {
                k -= num;
                -- max;
            }
        }
    }
}
```

#### [395. 至少有 K 个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

题目：

给你一个字符串 `s` 和一个整数 `k` ，请你找出 `s` 中的最长子串， 要求该子串中的每一字符出现次数都不少于 `k` 。返回这一子串的长度。



解答：

这题乍一看有点没有头绪，可能，我们还想用滑动窗口来解决，但是滑动窗口解决的是区间最小值问题，这是最大值，不太合适，所以我们可以试试使用回溯，回溯的意义在于：当某个区间不满足时，我们去寻找它的内部区间/旁侧区间，进行求解。这一题如果我们细细想想可以明白，如果这个区间内部没有这样的值，那肯定只能在它更小的内部区间，所以我们从整个字符串，慢慢缩小范围，满足则返回，不满足则以不满足的位置作为切分点，寻找子区间。或者这么理解，不满足的位置，其贡献值为负数，我们不可以选择它，而去选择它的两侧。

``` Java
class Solution {
    private int min;

    public int longestSubstring(String s, int k) {
        char[] str = s.toCharArray();
        min = k;
        return f(str, 0, str.length - 1);
    }

    // [ , ]
    private int f(char[] str, int from, int to) {
        if (to < from) {
            return 0;
        }
        if (to == from) {
            if (min == 1) {
                return 1;
            } else {
                return 0;
            }
        }
        int[] rcd = new int[256];
        int[] tmp = new int[256];
        Arrays.fill(rcd, 0);
        Arrays.fill(tmp, 0);
        // 统计每个字符出现字数
        for (int i = from; i <= to; ++ i) {
            rcd[str[i]] ++;
        }
        boolean allPassed = true;
        // 找到所有不够的
        for (int i ='a'; i < 'z'; ++ i) {
            if (rcd[i] > 0 && rcd[i] < min) {
                // System.out.println((char) i);
                allPassed = false;
                tmp[i] = 1;
            }
        }
        if (allPassed) {
            return to - from + 1;
        }
        int index = from;
        int ans = 0;
        for (int i = from; i <= to; ++ i) {
            if (tmp[str[i]] == 1) {
                // System.out.println(index + " : " + (i - 1));
                ans = Math.max(ans, f(str, index, i - 1));
                index = i + 1;
            }
        }
        ans = Math.max(ans, f(str, index, to));
        return ans;
    }
}
```

#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

题目：

给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)



解答：

直接使用DFS遍历一遍即可：

``` Java
class Solution {
    private int m;
    private int n;

    private int[][] input;
    private boolean[][] rcd;

    private static final int[] X = {0, 1, 0, -1};
    private static final int[] Y = {1, 0, -1, 0};

    public int maxAreaOfIsland(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        input = grid;
        rcd = new boolean[m][n];
        int ans = 0;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(rcd[i], false);
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int tmp = dfs(i, j);
                ans = Math.max(ans, tmp);
            }
        }
        return ans;
    }

    private int dfs(int x, int y) {
        if (!isInBound(x, y) || rcd[x][y] || input[x][y] == 0) return 0;
        int area = input[x][y];
        for (int k = 0; k < 4; ++ k) {
            int x0 = x + X[k];
            int y0 = y + Y[k];
            // 不再设置为false是为了实现单一路径标记，确保每个位置在全部遍历中只会走一次。
            rcd[x][y] = true;
            // 设置为false即可实现回溯，实现全部可能路径遍历，而不是只走一遍。
            int tmp = dfs(x0, y0);
            area += tmp;
        }
        return area;
    }

    private boolean isInBound(int x, int y) {
        return (x >= 0) && (y >= 0) && (x < m) && (y < n);
    }
}
```

#### [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

题目：

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

注意：字符串长度 和 k 不会超过 104。



解答：

如何实现替换次数相同时，替换后的相同字符长度最长，那肯定要以现在最长的字符为替换标准，这就涉及到区间更新找出现次数最多的字符，我们嗅到了滑动窗口的味道... ... 事实确实如此，我们通过滑动窗口，记录窗口内出现次数最多的字符是哪个，且记录它出现的次数，那我们需要替换的单词个数就是区间长度 - 最长重复字符出现的次数。动态更新时保证这个需要替换的值不会大于k即可。

``` Java
class Solution {
    public int characterReplacement(String s, int k) {
        char[] str = s.toCharArray();
        int len = str.length;
        int maxIndex = 0;
        int maxCount = 0;
        int needToEdit = 0;
        int l = 0, r = 0;
        int[] rcd = new int[256];
        int ans = 0;
        Arrays.fill(rcd, 0);
        while (r < len && l <= r) {
            rcd[str[r]]++;
            if (rcd[str[r]] >= maxCount) {
                maxCount = rcd[str[r]];
                maxIndex = str[r];
            }
            needToEdit = r - l + 1 - maxCount;
            // System.out.println("r: " + r + " ? " + l + " : " + r + " C " + maxCount);
            if (needToEdit <= k) {
                // System.out.println("r: " + r + " ? " + l + " : " + r + " S " + (r - l + 1));
                ans = Math.max(ans, (r - l + 1));
            } else if (needToEdit == k + 1) {
                rcd[str[l]] --;
                if (str[l] == maxIndex) maxCount --;
                ++ l;
                rcd[str[r]] --;
                -- r;
            }
            ++ r;
        }
        return ans;
    }
}
```

#### [524. 通过删除字母匹配到字典里最长单词](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

题目：

给你一个字符串 s 和一个字符串数组 dictionary 作为字典，找出并返回字典中最长的字符串，该字符串可以通过删除 s 中的某些字符得到。

如果答案不止一个，返回长度最长且字典序最小的字符串。如果答案不存在，则返回空字符串。



解答：

乍一看没啥头脑，还记得我们曾经讨论过的刷题经验吗？如果没有啥想法，或者有了好几个思路，但是拿捏不准时，可以试试直接模拟：

``` Java
class Solution {
    public String findLongestWord(String s, List<String> dictionary) {
        List<char[]> strings = dictionary.stream()
                .sorted((p1, p2) -> {
                    if (p1.length() != p2.length()) {
                        return p2.length() - p1.length();
                    } else {
                        return p1.compareTo(p2);
                    }
                })
                .map(String::toCharArray)
                .collect(Collectors.toList());
        int index = 0;
        int maxLen = 0;
        int maxIndex = -1;
        char[] origin = s.toCharArray();
        for (char[] chars : strings) {
            // 就硬匹配
            for (int i = 0; i < chars.length;) {
                for (int j = 0; j < origin.length;) {
                    if (i >= chars.length) {
                        break;
                    }
                    if (chars[i] == origin[j]) {
                        ++ i;
                    }
                    ++ j;
                }
                if (i == chars.length) {
                    if (chars.length > maxLen) {
                        maxLen = chars.length;
                        maxIndex = index;
                    }
                } else {
                    break;
                }
            }
            ++ index;
        }
        if (maxIndex == -1) {
            return "";
        }
        return new String(strings.get(maxIndex));
    }
}
```

#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

题目：

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。



解答：

既然是连续的，我们直接从连续序列前导数开始找，比如连续序列是x, x+1, x+2, ... ... 那我们就从x-1开始找，如果一个数不存在前导数，说明这个数肯定是序列起始位置，我们通过HashMap记录一下就可以快速判断前导数存不存在；然后每次+1进行找寻序列。

``` Java
class Solution {
    private final HashMap<Integer, Integer> parent = new HashMap<>();

    private int findParent(int index) {
        int tmpIndex = index;
        while (parent.get(index) != index) index = parent.get(index);
        parent.put(tmpIndex, index);
        return index;
    }

    private void merge(int from, int to) {
        parent.put(findParent(from), findParent(to));
    }

    public int longestConsecutive(int[] nums) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int a : nums) {
            parent.put(a, a);
            hashMap.put(a, 1);
        }
        int count = nums.length > 0 ? 1 : 0;
        for (int a : hashMap.keySet()) {
            if (hashMap.containsKey(a + 1)) {
                merge(a, a + 1);
            }
        }
        HashMap <Integer, Integer> rcd = new HashMap<>();
        for (int a : hashMap.keySet()) {
            int aa = findParent(a);
            rcd.merge(aa, 1, Integer::sum);
            count = Math.max(count, rcd.get(aa));
        }
        return count;
    }
}
```

#### [179. 最大数](https://leetcode-cn.com/problems/largest-number/)

题目：

给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

**注意：**输出结果可能非常大，所以你需要返回一个字符串而不是整数。



解答：

之所以贴这个题。是因为对于字符串字典序的排序，我们可以通过：s1+s2 ? s2+s1来进行比较。

``` Go
func largestNumber(nums []int) string {
	strs := make([]string, len(nums))
	for i, num := range nums {
		strs[i] = strconv.Itoa(num)
	}
	sort.Slice(strs, func(i, j int) bool {
		return strs[i]+strs[j] >= strs[j]+strs[i]
	})
	ans := strings.Join(strs, "")
	if ans[0] == '0' {
		return "0"
	}
	return ans
}
```

#### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

题目：

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。



解答：

这题有点像字典树，我们通过为每个节点定义26个子节点的形式，来表示这个节点可以延展出的单词。

``` Java
class Trie {

    private static class Node {
        Node[] children;
    }

    private final Node root;

    /** Initialize your data structure here. */
    public Trie() {
        root = new Node();
        root.children = new Node[26];
        for (int i = 0; i < 26; ++ i) {
            root.children[i] = null;
        }
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        insert(word, root);
    }

    private void insert(String word, Node parent) {
        if (word.length() == 1) {
            Node node = new Node();
            node.children = new Node[26];
            for (int i = 0; i < 26; ++ i) {
                node.children[i] = null;
            }
            parent.children[word.charAt(0) - 'a'] = node;
        } else if (parent.children[word.charAt(0) - 'a'] == null) {
            Node node = new Node();
            node.children = new Node[26];
            for (int i = 0; i < 26; ++ i) {
                node.children[i] = null;
            }
            parent.children[word.charAt(0) - 'a'] = node;
            insert(word.substring(1), parent.children[word.charAt(0) - 'a']);
        }
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        return search(word, root);
    }

    private boolean search(String word, Node node) {
        if (node == null) {
            return false;
        }
        if (word.length() == 1) {
            return node.children[word.charAt(0) - 'a'] != null;
        } else {
            if (node.children[word.charAt(0) - 'a'] != null) {
                return search(word.substring(1), node.children[word.charAt(1) - 'a']);
            } else {
                return false;
            }
        }
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        return search(prefix, root);
    }
}
```

#### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

题目：

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。



解答：

其实这题和前面的分割回文串很类似，我们依旧是寻找所有的切分点，但是这里需要我们找出所有分割的结果，那自然只能通过DFS进行子区间划分了。比方说我们发现from - i是一个回文串，那么我们可以利用深搜去找 i+1 - to的所有切分方案，然后整合，而i+1 - to的切分方案也可以通过这种方法处理。

``` Java
class Solution {
    private boolean[][] dp;

    private String string;

    public List<List<String>> partition(String s) {
        char[] str = s.toCharArray();
        string = s;
        dp = new boolean[str.length][str.length];
        for (int i = 0; i < str.length; ++ i) {
            for (int j = i; j < str.length; ++ j) {
                if (str[j] == str[i]) {
                    dp[i][j] = true;
                    int a = i + 1, b = j - 1;
                    while (a <= b) {
                        if (str[a] != str[b]) {
                            dp[i][j] = false;
                            break;
                        }
                        ++ a;
                        -- b;
                    }
                } else {
                    dp[i][j] = false;
                }
                // System.out.println(i + " = " + j + ": " + dp[i][j]);
            }
        }
        return f(0, str.length - 1);
    }

    private List<List<String>> f(int from, int to) {
        if (from > to) {
            return new ArrayList<>(0);
        } else if (from == to) {
            List<List<String>> ans = new ArrayList<>(1);
            List<String> result = new ArrayList<>(1);
            result.add(string.substring(from, from + 1));
            ans.add(result);
            return ans;
        } else {
            List<List<String>> ans = new ArrayList<>();
            for (int i = from; i <= to; ++ i) {
                if (dp[from][i]) {
                    List<List<String>> lists = f(i + 1, to);
                    String tmpStr = string.substring(from, i + 1);
                    // System.out.println(string.substring(from, to + 1) + ": " + tmpStr + " - " + lists);
                    for (var a : lists) {
                        List<String> tmp = new ArrayList<>();
                        tmp.add(tmpStr);
                        tmp.addAll(a);
                        ans.add(tmp);
                    }
                    if (lists.isEmpty()) {
                        List<String> tmp = new ArrayList<>();
                        tmp.add(tmpStr);
                        ans.add(tmp);
                    }
                }
            }
            return ans;
        }
    }
}
```

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

题目：

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。



解答：

很典型的背包题，背包容量是amout，寻找个数最小的一个组合：

``` Java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        Arrays.sort(coins);
        int[] dp = new int[10010];
        dp[0] = 0;
        for (int i = 1; i <= amount; ++ i) {
            dp[i] = Integer.MAX_VALUE;
            for (int a : coins) {
                if (i - a < 0) {
                    if (dp[i] == Integer.MAX_VALUE) {
                        dp[i] = -1;
                    }
                } else {
                    if (dp[i - a] != -1) {
                        dp[i] = Math.min(dp[i], dp[i - a]);
                    }
                }
                // System.out.println(i + " = " + dp[i]);
            }
            if (dp[i] == Integer.MAX_VALUE) {
                dp[i] = -1;
            }
            if (dp[i] != -1) {
                dp[i] += 1;
            }
        }
        return dp[amount];
    }
}
```

#### [1482. 制作 m 束花所需的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/)

题目：

给你一个整数数组 bloomDay，以及两个整数 m 和 k 。

现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。

花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好 可以用于 一束 花中。

请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。



解答：

我们需要找到某一个天数，所有盛开时间小于这个天数的花都是可以算进去的，而大于天数的则不可以，所以我们可以通过二分查找来找到这个合适的天数，至于判断天数合不合适，则是通过遍历数组，判断连续小于等于这个天数的花有多少个，如果有k个则记录，直到到达了m个为止。

``` Java
class Solution {
    public int minDays(int[] bloomDay, int m, int k) {
        if(m*k>bloomDay.length)return -1;
        int low=1,high=1;
        for(int i:bloomDay){
            high=Math.max(high,i);
        }
        while(low<high){
            int mid=low+(high-low)/2;
            if(isSuccess(bloomDay,mid,m,k)){
                high=mid;
            }else{
                low=mid+1;
            }
        }
        return high;
    }

    public boolean isSuccess(int[] bloomDay,int days,int m,int k){
        int sum=0,flowers=0;
        for(int i=0;i<bloomDay.length&&sum<m;i++){
            if(bloomDay[i]<=days){
                flowers++;
                if(flowers==k){
                    sum++;
                    flowers=0;
                }
            }else{
                flowers=0;
            }
        }
        return sum>=m;
    }
}
```

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

题目：

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。



解答：

区间最长，走滑窗：

``` Java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        char[] str = s.toCharArray();
        int len = str.length;
        boolean[] rcd = new boolean[256];
        Arrays.fill(rcd, false);
        int ans = 0;
        int l = 0, r = 0;
        while (l <= r && r < len) {
            if (!rcd[str[r]]) {
                rcd[str[r]] = true;
                ans = Math.max(ans, r - l + 1);
                ++ r;
            } else {
                while (str[l] != str[r]) {
                    rcd[str[l]] = false;
                    ++ l;
                }
                rcd[str[l]] = false;
                ++ l;
            }
        }
        return ans;
    }
}
```

#### [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

题目：

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：

 nums1[i] == nums2[j]
且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。



解答：

简单DP处理即可。

``` Java
class Solution {
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        if (nums1.length == 1 && nums2.length == 1) {
            return nums1[0] == nums2[0] ? 1 : 0;
        } else if (nums1.length == 1) {
            for (int i = 0; i < nums2.length; ++ i) {
                if (nums2[i] == nums1[0]) {
                    return 1;
                }
            }
            return 0;
        } else if (nums2.length == 1) {
            for (int i = 0; i < nums1.length; ++ i) {
                if (nums1[i] == nums2[0]) {
                    return 1;
                }
            }
            return 0;
        }
        int[][] dp = new int[nums1.length][nums2.length];
        for (int i = 0; i < nums1.length; ++ i) {
            for (int j = 0; j < nums2.length; ++ j) {
                if (i >= 1 && j >= 1) {
                    if (nums1[i] == nums2[j]) {
                        dp[i][j] = dp[i-1][j-1] + 1;
                    } else {
                        dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                    }
                } else if (i >= 1) {
                    if (nums1[i] == nums2[j]) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i-1][j];
                    }
                } else if (j >= 1) {
                    if (nums1[i] == nums2[j]) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i][j-1];
                    }
                } else {
                    dp[0][0] = nums1[0] == nums2[0] ? 1 : 0;
                }
            }
        }
        return dp[nums1.length-1][nums2.length-1];
    }
}
```

#### [1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

题目：

给出一个字符串 s（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中 不应 包含任何括号。



解答：

涉及到左右匹配问题的，我们可以使用栈来解决：

``` Java
class Solution {
    public String reverseParentheses(String s) {
        Stack<StringBuilder> stack = new Stack<>();
        stack.push(new StringBuilder());
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                StringBuilder buffer = new StringBuilder();
                stack.push(buffer);
            } else if (s.charAt(i) == ')') {
                StringBuilder pop = stack.pop();
                StringBuilder reverse = pop.reverse();
                stack.peek().append(reverse);
            } else {
                stack.peek().append(s.charAt(i));
            }
        }
        return stack.peek().toString();
    }
}
```

#### [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

题目：

根据 逆波兰表示法，求表达式的值。

有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

 

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。



解答：

使用栈模拟一下运算顺序即可：

``` Java
class Solution {
    private final LinkedList<Integer> nums = new LinkedList<>();

    public int evalRPN(String[] tokens) {
        for (String s : tokens) {
            if (s.length() == 1 && (s.equals("+") || s.equals("-") || s.equals("*") || s.equals("/"))) {
                int a = nums.getFirst();
                nums.removeFirst();
                int b = nums.getFirst();
                nums.removeFirst();
                int c = 0;
                switch (s) {
                    case "*":
                        c = b * a;
                        break;
                    case "/":
                        c = b / a;
                        break;
                    case "+":
                        c = b + a;
                        break;
                    case "-":
                        c = b - a;
                }
                nums.addFirst(c);
            } else {
                nums.addFirst(Integer.parseInt(s));
            }
        }
        return nums.getFirst();
    }
}
```

#### [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)

题目：

给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：

子数组大小 至少为 2 ，且
子数组元素总和为 k 的倍数。
如果存在，返回 true ；否则，返回 false 。

如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。0 始终视为 k 的一个倍数。



解答：

我们第一反应是循环，或者使用sum\[]来记录，但是涉及到的时间复杂度可能都不尽人意。那我们来看，既然是要求整数倍，那是不是可以这样想，如果sum\[j] % k == a同时sum\[i] & k == a；那么i+1 - j之间的和就肯定是k的倍数了，因为sum\[j] - sum\[i] == i+1 - j的和，又因为拥有同一个余数的原因，所以可以消掉。于是这题成了遍历一遍的数学题，接下来使用HashMap记录一下即可。

``` Java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        if (nums.length < 2) return false;
        for (int i = 0; i < nums.length-1; ++i) 
            if(nums[i] == 0 && nums[i+1] == 0) return true;
        if (k == 0) return false;
        if (k < 0) k = -k;
        
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int sum = 0;
        for (int i = 0; i < nums.length; ++ i) {
            sum += nums[i];
            int mod = sum % k;
            if (map.containsKey(mod)) {
                if(i-map.get(mod) > 1)
                    return true;
            }
            else // 不存在再更新
                map.put(mod, i);
        }
        return false;
    }
}
```

#### [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)

题目：

给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。



解答：

和前面那题有着异曲同工之妙，依旧是前缀和，依旧是sum\[]数组。只不过这次我们记录一下1或0出现的次数，作为sum\[]值，然后使用HashMap处理：

``` Java
class Solution {
    public int findMaxLength(int[] nums) {
        int maxLength = 0;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int counter = 0;
        map.put(counter, -1);
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num == 1) {
                counter++;
            } else {
                counter--;
            }
            if (map.containsKey(counter)) {
                int prevIndex = map.get(counter);
                maxLength = Math.max(maxLength, i - prevIndex);
            } else {
                map.put(counter, i);
            }
        }
        return maxLength;
    }
}
```

#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

题目：

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。



解答：

内循环DP记录一下，从可以切分的位置进行判断后面的位置能否拆分：

``` Java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> hashSet = new HashSet<>();
        hashSet.addAll(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 0; i < s.length(); ++ i) {
            for (int j = i; j >= 0; -- j) {
              	// 内循环判断
                if (dp[j]) {
                    if (hashSet.contains(s.substring(j, i + 1))) {
                        dp[i + 1] = true;
                        break;
                    }
                }
            }
        }
        return dp[s.length()];
    }
}
```

#### [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

题目：

给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。



解答：

01背包问题的伪装罢了，不过我们此时要注意，限制因素变成了两个，一个是0的数量，一个是1的数量，所以我们需要注意这一点：

``` Java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        dp[0][0] = 0;
        for (String s : strs) {
            int[] zeroAndOne = calcZeroAndOne(s);
            int zeros = zeroAndOne[0];
            int ones = zeroAndOne[1];
            for (int i = m; i >= zeros; i--) {
                for (int j = n; j >= ones; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeros][j - ones] + 1);
                }
            }
        }
        return dp[m][n];
    }

    private int[] calcZeroAndOne(String str) {
        int[] res = new int[2];
        for (char c : str.toCharArray()) {
            res[c - '0']++;
        }
        return res;
    }
}
```

#### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

题目：

给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。



解答：

因为只能添加`'+'`和`'-'`两种符号，所以最后我们整个表达式肯定是a - (sum-a) = target，其中sum为数组总和。化简一下得a = (sum + target) / 2；我们让它为k，如果(sum + target)不是2的倍数，那么无论如何都凑不出这样的表达式。



所以最后我们只要找一个子集，使得子集的和等于k即可。而这又回到了背包问题，还是01背包。

``` Java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        // 关键在于理解，本题就是要找一个子集，满足子集之和=(sum(nums) + target) / 2;
        int sum = 0;
        for (int a : nums) {
            sum += a;
        }
        if ((sum + target) % 2 != 0) {
            return 0;
        }
        int k = (sum + target) / 2;
        int[] dp = new int[k + 1];
        dp[0] = 1;
        for (int a : nums) {
            for (int i = k; i >= a; -- i) {
                // 核心代码，一维DP
                dp[i] += dp[i-a];
            }
        }
        return dp[k];
    }
}
```

#### [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

题目：

有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。



解答：

这题和前面一题有着异曲同工之妙，我们可以想一下，石头碾碎最后的结果，一定是a - (sum - a) == 2a - sum，我们只要求得子集的二倍，并且让它尽可能接近sum即可。可以使用背包进行存在性判断，也可以直接求：

``` Java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int i = 0; i < stones.length; ++ i) {
            sum += stones[i];
            stones[i] *= 2;
        }
        int[][] dp = new int[stones.length][sum + 1];
        for (int i = stones[0]; i <= sum; ++ i) dp[0][i] = stones[0];
        for (int i = 1; i < stones.length; ++ i) {
            for (int j = 0; j <= sum; ++ j) {
                if (j >= stones[i]) {
                    // 在总和不超过sum的情况下；不选所能求得的最大值和选了所能求得的最大值之间进行比较
                    dp[i][j] = Math.max(dp[i-1][j], dp[i-1][j-stones[i]] + stones[i]);
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return sum - dp[stones.length-1][sum];
    }
}
```

#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

题目：

给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。



解答：

完全背包问题：

``` Java
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount+1];
        dp[0] = 1;
        for (int i = 1; i <= coins.length; ++ i) {
            for (int j = 0; j <= amount; ++ j) {
                if (j >= coins[i-1]) {
                    dp[j] = dp[j] + dp[j-coins[i-1]];
                }
            }
        }
        return dp[amount];
    }
}
```

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

题目：

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。



解答：

完全背包问题：

``` Java
class Solution {
    public int numSquares(int n) {
        int[] rcd = new int[n+1];
        int size = 0;
        for (int i = 1; i * i <= n; ++ i) {
            rcd[i-1] = i*i;
            size++;
        }
        int[][] dp = new int[size+1][n+1];
        for (int i = 0; i <= n; ++ i) {
            dp[0][i] = i;
        }
        dp[0][0] = 0;
        for (int i = 1; i <= size; ++ i) {
            for (int j = 0; j <= n; ++ j) {
                if (j < rcd[i-1]) {
                    dp[i][j] = dp[i-1][j];
                } else {
                    dp[i][j] = Math.min(dp[i-1][j], dp[i][j-rcd[i-1]]+1);
                }
                // System.out.print(dp[i][j] + " ");
            }
            // System.out.println();
        }
        return dp[size][n];
    }
}
```

#### [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

题目：

给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。



解答：

模拟题，但是比较经典：

``` Java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        int size = 0;
        ListNode tmp = head;
        ListNode prev = tmp;
        while (tmp != null) {
            ++ size;
            prev = tmp;
            tmp = tmp.next;
        }
        if (size == 0) {
            return head;
        }
        // System.out.println(size);
        prev.next = head;
        tmp = head;
        k %= size;
        for (int i = 0; i < size-k; ++ i) {
            prev = tmp;
            tmp = tmp.next;
        }
        // System.out.println(tmp.val);
        prev.next = null;
        return tmp;
    }
}
```

#### [1239. 串联字符串的最大长度](https://leetcode-cn.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/)

题目：

给定一个字符串数组 arr，字符串 s 是将 arr 某一子序列字符串连接所得的字符串，如果 s 中的每一个字符都只出现过一次，那么它就是一个可行解。

请返回所有可行解 s 中最长长度。



解答：

回溯处理一下就行：

``` Java
class Solution {

    private String[] ss;

    private int ans;

    private boolean[] marked;

    public int maxLength(List<String> arr) {
        ss = arr.toArray(new String[0]);
        String[] ss0 = new String[ss.length];
        int index = 0;
        marked = new boolean[26];
        for (String s : ss) {
            boolean flag = false;
            for (char c : s.toCharArray()) {
                if (marked[c - 'a']) {
                    flag = true;
                    break;
                }
                marked[c - 'a'] = true;
            }
            if (!flag) ss0[index++] = s;
            Arrays.fill(marked, false);
        }
        ss = new String[index];
        System.arraycopy(ss0, 0, ss, 0, index);
        dfs(0, new StringBuilder());
        return ans;
    }

    private void dfs(int i, StringBuilder stringBuilder) {
        if (i >= ss.length) {
            // System.out.println(stringBuilder.toString());
            ans = Math.max(stringBuilder.length(), ans);
            return ;
        }
        boolean flag = false;
        for (char c : ss[i].toCharArray()) {
            if (marked[c-'a']) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            for (char c : ss[i].toCharArray()) {
                marked[c-'a'] = true;
            }
            dfs(i+1, new StringBuilder(stringBuilder).append(ss[i]));
            for (char c : ss[i].toCharArray()) {
                marked[c-'a'] = false;
            }
        }
        dfs(i+1, stringBuilder);
    }
}
```

#### [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

题目：

运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？



解答：

直接使用LinkedHashMap模拟即可。

``` Java
class LRUCache {

    private final int cap;

    private final LinkedHashMap<Integer, Integer> linkedHashMap;

    public LRUCache(int capacity) {
        this.cap = capacity;
        this.linkedHashMap = new LinkedHashMap<>(capacity, 0.75f, true);
    }
    
    public int get(int key) {
        if (linkedHashMap.keySet().contains(key)) {
            return linkedHashMap.get(key);
        } else {
            return -1;
        }
    }

    public void put(int key, int value) {
        if (!linkedHashMap.containsKey(key)) {
            if (linkedHashMap.size() == this.cap) {
                linkedHashMap.remove(linkedHashMap.keySet().iterator().next());
            }
        }
        linkedHashMap.put(key, value);
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```

#### [930. 和相同的二元子数组](https://leetcode-cn.com/problems/binary-subarrays-with-sum/)

题目：

给你一个二元数组 `nums` ，和一个整数 `goal` ，请你统计并返回有多少个和为 `goal` 的 **非空** 子数组。

**子数组** 是数组的一段连续部分。



解答：

使用前缀和与HashMap进行记录即可。

``` Java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        Map<Integer,Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0;
        int res = 0;
        for (int i=0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - goal)) {
                // 更新总数
                res += map.get(sum - goal);
            }
            // 更新sum出现的次数
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
}
```

#### [981. 基于时间的键值存储](https://leetcode-cn.com/problems/time-based-key-value-store/)

题目：

设计一个基于时间的键值数据结构，该结构可以在不同时间戳存储对应同一个键的多个值，并针对特定时间戳检索键对应的值。

实现 TimeMap 类：

TimeMap() 初始化数据结构对象
void set(String key, String value, int timestamp) 存储键 key、值 value，以及给定的时间戳 timestamp。
String get(String key, int timestamp)
返回先前调用 set(key, value, timestamp_prev) 所存储的值，其中 timestamp_prev <= timestamp 。
如果有多个这样的值，则返回对应最大的  timestamp_prev 的那个值。
如果没有值，则返回空字符串（""）。



解答：

通过优先队列实现排序，找到所有时间上大于给定时间戳timestamp的键值对：

``` Java
class TimeMap {

        private final HashMap<String, PriorityQueue<Pair>> hashMap = new HashMap<>();

        /** Initialize your data structure here. */
        public TimeMap() {
            ;
        }

        public void set(String key, String value, int timestamp) {
            Pair pair = new Pair();
            pair.value = value;
            pair.timestamp = timestamp;
            if (!hashMap.containsKey(key)) {
                hashMap.put(key, new PriorityQueue<>());
            }
            hashMap.get(key).add(pair);
        }

        public String get(String key, int timestamp) {
            LinkedList<Pair> pairs = new LinkedList<>();
            PriorityQueue<Pair> priorityQueue = hashMap.get(key);
            if (priorityQueue == null) {
                return "";
            }
            while (!priorityQueue.isEmpty()) {
                if (priorityQueue.peek().timestamp <= timestamp) {
                    break;
                } else {
                    pairs.addLast(priorityQueue.poll());
                }
            }
            String ans = priorityQueue.isEmpty() ? "" : priorityQueue.peek().value;
            while (!pairs.isEmpty()) {
                priorityQueue.add(pairs.getLast());
                pairs.removeLast();
            }
            return ans;
        }

        private static class Pair implements Comparable<Pair> {
            int timestamp;
            String value;

            @Override
            public int compareTo(Pair o) {
                return -1 * Integer.compare(this.timestamp, o.timestamp);
            }
        }
    }

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap obj = new TimeMap();
 * obj.set(key,value,timestamp);
 * String param_2 = obj.get(key,timestamp);
 */
```

#### [1818. 绝对差值和](https://leetcode-cn.com/problems/minimum-absolute-sum-difference/)

题目：

给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。

数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。

你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。

在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对 109 + 7 取余 后返回。

|x| 定义为：

如果 x >= 0 ，值为 x ，或者
如果 x <= 0 ，值为 -x



解答：

这一题我们乍一看，没什么头绪，其实思路大家都知道，就是比较nums2\[i]与nums1\[i]的差值的变化幅度，找到最大的一个，就是我们需要替换的位置，至于新的差值则是和[0, i-1], [i+1, nums1.length-1]之间的比较。



那怎么快速找到我们需要的值呢？答案是二分，找到排序后的nums1里最接近nums2\[i]的值即可，这里我们可以使用红黑树完成快速找到最接近的值的操作：

``` Java
class Solution {
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        long ans = 0;
        long mod = 1000000007;
        TreeSet<Integer> treeSet = new TreeSet<>();
        for (int i = 0; i < nums2.length; ++ i) {
            ans += Math.abs(nums1[i] - nums2[i]);
            treeSet.add(nums1[i]);
        }
        long diff = 0;
        for (int i = 0; i < nums2.length; ++ i) {
            Integer a;
            int tmp1 = Math.abs(((a = treeSet.floor(nums2[i])) == null ? nums1[i] : a) - nums2[i]);
            int tmp2 = Math.abs(((a = treeSet.ceiling(nums2[i])) == null ? nums1[i] : a) - nums2[i]);
            int tmp0 = Math.min(tmp1, tmp2);
            int tmp = Math.abs(nums2[i] - nums1[i]);
            if (tmp - tmp0 > diff) {
                diff = tmp-tmp0;
            }
        }
        ans -= diff;
        return (int) (ans % mod);
    }
}
```

#### [1846. 减小和重新排列数组后的最大元素](https://leetcode-cn.com/problems/maximum-element-after-decreasing-and-rearranging/)

题目：

给你一个正整数数组 arr 。请你对 arr 执行一些操作（也可以不进行任何操作），使得数组满足以下条件：

arr 中 第一个 元素必须为 1 。
任意相邻两个元素的差的绝对值 小于等于 1 ，也就是说，对于任意的 1 <= i < arr.length （数组下标从 0 开始），都满足 abs(arr[i] - arr[i - 1]) <= 1 。abs(x) 为 x 的绝对值。
你可以执行以下 2 种操作任意次：

减小 arr 中任意元素的值，使其变为一个 更小的正整数 。
重新排列 arr 中的元素，你可以以任意顺序重新排列。
请你返回执行以上操作后，在满足前文所述的条件下，arr 中可能的最大值 。



解答：

计数排序问题。因为最大值肯定不会大于数组长度，最理想的情况就是arr\[i] == i+1，所以对于不满足的直接“挤掉”，然后用后面的进行替换：

``` Java
class Solution {
  	// 这代码不是我写的，看来是为了打开抄了一位老哥的，蛮牛的
    public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
        // 显然最大值不会超过数组长度, 更大的值必须舍弃
        int n = arr.length;
        int[] count = new int[n + 1]; // 计数
        // 为什么长度是 n + 1 呢? 因为长度 n + 1 => 最大值为n, 再大照样会被缩小
        for (int i = 0; i < n; i++) {
            count[Math.min(arr[i], n)]++; // 使用min函数 => 避免数组溢出
        }
        int res = 0;
        for (int i = 1; i <= n; i++) { // 根据键统计值
            res = Math.min(res + count[i], i); // 对于修改后的 arr, 其每个元素必须等于 min(前数 + 1, 后数), 以满足元素只能变小的限制
        }
        // 为什么是count[i], 因为将重复的元素i都合并处理了, 加了 count[i] 个 1
        return res;
    }
}
```

#### [面试题 10.02. 变位词组](https://leetcode-cn.com/problems/group-anagrams-lcci/)

题目：

编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

**注意：**本题相对原题稍作修改



解答：

对字符串的字符数组重新排序，进行合并即可，而不需要进行全排列，说白了就是字符串并查集的处理。

``` Java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> modeWords = new HashMap<String, List<String>> ();
        for (String word : strs) {
            char[] tmp = word.toCharArray();
            Arrays.sort(tmp);
            String mode = new String(tmp);
            List<String> words = modeWords.getOrDefault(mode, new ArrayList<String>());
            words.add(word);
            modeWords.put(mode, words);
        }
        return new ArrayList<List<String>> (modeWords.values());
    }
}
```

#### [1838. 最高频元素的频数](https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/)

题目：

元素的 频数 是该元素在一个数组中出现的次数。

给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。

执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。



解答：

滑动窗口题，因为只能增加，所以先对数组排序，然后每次以最右边的元素做为标准进行统计操作次数，同时记得右滑即可。

``` Java
class Solution {
    public int maxFrequency(int[] nums, int k) {
        int n = nums.length;
        Arrays.sort(nums);
        int l = 0,r = 1;
        int max = 1; //最大频数
        long total = 0; //操作数
        while(r < n){
            /* r从1开始，比如1，4，8，13，此时是4，需要把1变成当前数字nums[r]，需要的次数则是1*3次
            此时total所代表的数都已经变成了4，此时遍历下一位8，因为前面的数字都是相等的，所以通过计算
            (r-l)*(nums[r]-nums[r-1])即可计算新的total值
            但是每次计算total后都要检查是否超过k，如果超过了k则需要不断从total中减去nums[l]的值
            为什么是减去nums[l]的值呢？
            数组是有序的，从后一个数变成更大的数x代价一定要比前一个数变成x的代价小，所以这里需要不断减去
            nums[l]的值更新total以使得total重新 <= k,因为max会记录每次最大的频数，也不用担心会漏掉
            */
            total += (r-l)*(nums[r]-nums[r-1]);
            while(total > k && l < r){
                total -= nums[r]-nums[l++];
            }
            max = Math.max(max,r-l+1);
            r++;
        }
        return max;
    }
}
```

#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

题目：

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。



解答：

经典题，直接上二分法的：

``` Java
class Solution {
    
    // 保证在取得相同LIS长度的情况下，可以得到上升度最缓的一个，即dp可以保证在长度相同的情况下，末尾是最小的
    // 换句话说，dp[len]保证在取到长度为len的所有LIS的情况下，可以得到末尾最小的那个LIS。
    // 看一个例子：1 3 5 9 2 4 6 7，加入我们此时已经得到了：1 3 5 9，然后替换同等长度的得到：1 2 4 6 7，此时可以取得7，于是长度得到了更新。
    public int lengthOfLIS(int[] nums) {
        int len = 1;
        int[] dp = new int[nums.length + 1];
        dp[len] = nums[0];
        for (int i = 1; i < nums.length; ++ i) {
            if (dp[len] >= nums[i]) {
                int index = Arrays.binarySearch(dp, 1, len+1, nums[i]);
                if (index < 0) {
                    index = (-index) - 1;
                    dp[index] = nums[i];
                }
            } else {
                dp[++len] = nums[i];
            }
        }
        return len;
    }
}
```

#### [743. 网络延迟时间](https://leetcode-cn.com/problems/network-delay-time/)

题目：

有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。



解答：

最短路跑一遍，找到所有最短路中最长的一个即可：

``` Java
class Solution {
    public int networkDelayTime(int[][] times, int N, int K) {
        // 构建邻接表，用于存放各个点到各个点的距离
        int[][] graph = new int[N + 1][N + 1];
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                graph[i][j] = -1;
            }
        }
        // 遍历times填充邻接表
        for (int[] time : times) {
            graph[time[0]][time[1]] = time[2];
        }

        // 存放 K 到各个点的最短路径，最大的那个最短路径即为结果
        int[] distance = new int[N + 1];
        Arrays.fill(distance, -1);

        // 初始化 distance 为 K 到各个节点的距离
        for (int i = 1; i <= N; i++) {
            distance[i] = graph[K][i];
        }
        // K到达K本身的节点初始化为 0
        distance[K] = 0;

        // 判断是否找到K到达该点最短路径
        boolean[] visited = new boolean[N + 1];
        visited[K] = true;

        // 遍历除K本身节点之外的所有N-1个节点
        for (int i = 1; i <= N - 1; i++) {
            int minDistance = Integer.MAX_VALUE;
            int minIndex = 1;
            // 遍历所有节点，找到离K最近的节点
            for (int j = 1; j <= N; j++) {
                if (!visited[j] && distance[j] != -1 && distance[j] < minDistance) {
                    minDistance = distance[j];
                    minIndex = j;
                }
            }

            // 标记最近距离节点找到
            visited[minIndex] = true;

            // 根据刚刚找到的最短距离节点，通过该节点更新K节点与其他的节点的距离
            for (int j = 1; j <= N; j++) {
                // 如果已更新的最短节点可以到达当前节点
                if (graph[minIndex][j] != -1) {
                    if (distance[j] != -1) {
                        // 取之前路径与当前更新路径的最小值
                        distance[j] = Math.min(distance[j], distance[minIndex] + graph[minIndex][j]);
                    } else {
                        // 该节点是第一次访问，直接更新
                        distance[j] = distance[minIndex] + graph[minIndex][j];
                    }
                }
            }
        }

        int maxDistance = 0;
        // 遍历最大值，如果有节点未被访问，返回 -1，否则返回最大最短路径
        for (int i = 1; i <= N; i++) {
            if (distance[i] == -1) {
                return -1;
            }
            maxDistance = Math.max(distance[i], maxDistance);
        }

        return maxDistance;
    }
}
```

#### [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

题目：给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。

两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
提示：a + b 意味着字符串 a 和 b 连接。



**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)



解答：

双串问题=DP问题，直接硬DP就行：

``` Java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        char[] str1 = s1.toCharArray();
        char[] str2 = s2.toCharArray();
        char[] str = s3.toCharArray();
        boolean[][] dp = new boolean[str1.length+1][str2.length+1];
        if (str.length != str1.length + str2.length) {
            return false;
        }
        dp[0][0] = true;
        for (int i = 1; i <= str1.length; ++ i) {
            dp[i][0] = str[i-1] == str1[i-1] && dp[i-1][0];
        }
        for (int i = 1; i <= str2.length; ++ i) {
            dp[0][i] = str2[i-1] == str[i-1] && dp[0][i-1];
        }
        for (int i = 1; i <= str1.length; ++ i) {
            for (int j = 1; j <= str2.length; ++ j) {
                if ((dp[i][j-1] && str2[j-1] == str[i+j-1]) || (dp[i-1][j] && str1[i-1] == str[i+j-1])) {
                    dp[i][j] = true;
                } else {
                    dp[i][j] = false;
                }
            }
        }
        return dp[str1.length][str2.length];
    }
}
```

#### [802. 找到最终的安全状态](https://leetcode-cn.com/problems/find-eventual-safe-states/)

题目：

在有向图中，以某个节点为起始节点，从该点出发，每一步沿着图中的一条有向边行走。如果到达的节点是终点（即它没有连出的有向边），则停止。

对于一个起始节点，如果从该节点出发，无论每一步选择沿哪条有向边行走，最后必然在有限步内到达终点，则将该起始节点称作是 安全 的。

返回一个由图中所有安全的起始节点组成的数组作为答案。答案数组中的元素应当按 升序 排列。

该有向图有 n 个节点，按 0 到 n - 1 编号，其中 n 是 graph 的节点数。图以下述形式给出：graph[i] 是编号 j 节点的一个列表，满足 (i, j) 是图的一条有向边。



**示例 1：**

![Illustration of graph](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/03/17/picture1.png)





解答：

本质就是找有向环图，我们使用拓扑排序跑一遍，所有的入度不为0的节点，说明存在环，记录一下输出即可。



这里我说一下拓扑排序：

* 1⃣️找到所有入度为0的节点，入队。
* 2⃣️弹出一个节点。
* 3⃣️遍历它所能到达的所有节点，把这些节点入度-1。
* 4⃣️如果-1之后某个节点入度为0，则入队。
* 5⃣️重复4⃣️。

``` Java
class Solution {
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        int[] rd = new int[n];
        List<Integer>[] reverseGraph = new ArrayList[n];
        List<Integer> res = new ArrayList<>();
        Deque<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            reverseGraph[i] = new ArrayList<Integer>();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < graph[i].length; j++) {
                reverseGraph[graph[i][j]].add(i);
            }
            rd[i] = graph[i].length;
        }
       
        for (int i = 0; i < n; i++){
            if(rd[i] == 0){
                queue.offer(i);
            }
        }
        
        while (!queue.isEmpty()) {
            int e = queue.poll();
            for (int x : reverseGraph[e]) {
                if (--rd[x] == 0) {
                    queue.offer(x);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (rd[i] == 0) {
                res.add(i);
            }
        }
        return res;
    }
}
```

#### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

题目：

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。



解答：

双串问题=DP问题：

``` Java
class Solution {
    public int longestPalindromeSubseq(String s) {
        char[] str = s.toCharArray();
        int[][] dp = new int[str.length][str.length];
        for (int i = 0; i < str.length; ++ i) {
            dp[i][i] = 1;
        }
        for (int i = str.length - 2; i >= 0; -- i) {
            for (int j = i + 1; j < str.length; ++ j) {
                if (str[i] == str[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][str.length - 1];
    }
}
```

#### [576. 出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)

题目：

给你一个大小为 m x n 的网格和一个球。球的起始坐标为 [startRow, startColumn] 。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。你 最多 可以移动 maxMove 次球。

给你五个整数 m、n、maxMove、startRow 以及 startColumn ，找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对 109 + 7 取余 后的结果。



解答：

第一反应是DFS，后来觉得DP应该跑得更快，这题完全可以用状态转移求得：

``` Java
class Solution {

    private int[] X = new int[]{1, 0, -1, 0};

    private int[] Y = new int[]{0, 1, 0, -1};
    
    private int m, n;

    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        this.m = m;
        this.n = n;
        long[][][] dp = new long[m+1][n+1][maxMove+1];
        for (int k = 1; k <= maxMove; ++ k) {
            for (int i = 0; i < m; ++ i) {
                for (int j = 0; j < n; ++ j) {
                    for (int index = 0; index < 4; ++ index) {
                        int a = i + X[index];
                        int b = j + Y[index];
                        if (!isBound(a, b)) {
                            dp[i][j][k] += 1;
                        } else {
                            dp[i][j][k] += dp[a][b][k-1];
                        }
                        dp[i][j][k] %= 1000000007;
                    }
                }
            }
        }
        return (int) dp[startRow][startColumn][maxMove];
    }

    private boolean isBound(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }
}
```

#### [71. 简化路径](https://leetcode-cn.com/problems/simplify-path/)

题目：

给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为更加简洁的规范路径。

在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，'//'）都被视为单个斜杠 '/' 。 对于此问题，任何其他格式的点（例如，'...'）均被视为文件/目录名称。

请注意，返回的 规范路径 必须遵循下述格式：

始终以斜杠 '/' 开头。
两个目录名之间必须只有一个斜杠 '/' 。
最后一个目录名（如果存在）不能 以 '/' 结尾。
此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
返回简化后得到的 规范路径 。



解答：

模拟题，涉及到当前目录到父级目录，可以使用栈来模拟：

``` Java
class Solution {

    public String simplifyPath(String path) {
        String[] s = path.split("/");
        LinkedList<String> stack = new LinkedList<>();
        for (String str : s) {
            if (isParent(str)) {
                stack.pollFirst();
            } else if (isNow(str)) {
                ;
            } else {
                if ("".equals(str)) {
                    ;
                } else {
                    stack.addFirst(str);
                }
            }
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("/");
        while (!stack.isEmpty()) {
            stringBuilder.append(stack.removeLast()).append("/");
        }
        if (stringBuilder.length() > 1) {
            stringBuilder.deleteCharAt(stringBuilder.length()-1);
        }
        return stringBuilder.toString();
    }

    private boolean isParent(String str) {
        return str.equals("..") || str.equals("../");
    }

    private boolean isNow(String str) {
        return str.equals(".") || str.equals("./");
    }
}
```

#### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

题目：

实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。



解答：

本题比较经典，可以好好思考一下。



使用贪心思想，想要得到尽可能小的增加，就要尽可能靠右进行值交换，这里说一下，全排列本质就是把前面小的值和后面大的值交换。所以我们尽可能让交换的左边靠右，同时与它交换的值必须大于它，同时尽可能小。

``` Java
class Solution {

    public void nextPermutation(int[] nums) {
        int minL = -1, minR = -1;
        for (int i = 0; i < nums.length; ++ i) {
            int maxRight = Integer.MAX_VALUE;
            for (int j = i + 1; j < nums.length; ++ j) {
              	// 找到右边比nums[i]大的值
                if (nums[j] > nums[i]) {
                  	// 尽可能向右更新
                    if (i > minL) {
                        minL = i;
                    }
                  	// 尽可能使用更小的值更新
                    if (nums[j] < maxRight) {
                        maxRight = nums[j];
                        minR = j;
                    }
                }
            }
        }
        if (minL == -1 && minR == -1) {
            Arrays.sort(nums);
        } else {
            int tmp = nums[minL];
            nums[minL] = nums[minR];
            nums[minR] = tmp;
            Arrays.sort(nums, minL + 1, nums.length);
        }
    }
}
```

#### [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)

题目：

给你一个 **无重叠的** *，*按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。



解答：

题目本身不难，是一道中规中矩的模拟题，但是细节繁多，处理起来比较麻烦，此外，为了方便处理，我在原本的数组前后各添加了一个边界区间，最后记得去掉即可。

``` Java
class Solution {

    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals.length == 0) {
            return new int[][] {{newInterval[0], newInterval[1]}};
        }
        if (newInterval[1] < intervals[0][0]) {
            int[][] tmp = new int[intervals.length+1][2];
            System.arraycopy(intervals, 0, tmp, 1, intervals.length);
            tmp[0] = newInterval;
            return tmp;
        }
        if (newInterval[0] > intervals[intervals.length-1][1]) {
            int[][] tmp = new int[intervals.length+1][2];
            System.arraycopy(intervals, 0, tmp, 0, intervals.length);
            tmp[intervals.length] = newInterval;
            return tmp;
        }
        if (newInterval[0] <= intervals[0][0] && newInterval[1] >= intervals[intervals.length-1][1]) {
            return new int[][] {{newInterval[0], newInterval[1]}};
        }
        int[][] tmp = new int[intervals.length+2][2];
        System.arraycopy(intervals, 0, tmp, 1, intervals.length);
        tmp[0][0] = Integer.MIN_VALUE;
        tmp[0][1] = Integer.MIN_VALUE+1;
        tmp[tmp.length-1][0] = Integer.MAX_VALUE-1;
        tmp[tmp.length-1][1] = Integer.MAX_VALUE;
        int l = -1, r = -1;
        for (int i = 0; i < tmp.length-1; ++ i) {
            if (tmp[i][0] <= newInterval[0] && newInterval[0] < tmp[i+1][0]) {
                l = i;
            }
            if (tmp[i][1] < newInterval[1] && newInterval[1] <= tmp[i+1][1]) {
                r = i;
            }
        }
        int[][] ansTmp = new int[tmp.length+1][2];
        int idx = 0;
        for (int i = 0; i < tmp.length; ++ i) {
            if (i == l) {
                i = r;
                if (newInterval[0] <= tmp[l][1]) {
                    ansTmp[idx][0] = tmp[l][0];
                } else {
                    ansTmp[idx][0] = tmp[l][0];
                    ansTmp[idx][1] = tmp[l][1];
                    ++ idx;
                    ansTmp[idx][0] = newInterval[0];
                }
                if (newInterval[1] >= tmp[r+1][0]) {
                    ansTmp[idx][1] = tmp[r+1][1];
                    ++ i;
                    ++ idx;
                } else {
                    ansTmp[idx][1] = newInterval[1];
                    ++ idx;
                }
                continue;
            }
            ansTmp[idx] = tmp[i];
            ++ idx;
        }
        int from = 0, end = 0;
        for (int i = 0; i < ansTmp.length; ++ i) {
            if (ansTmp[i][0] == Integer.MIN_VALUE) {
                from = i;
                break;
            }
        }
        for (int i = ansTmp.length-1; i >= 0; -- i) {
            if (ansTmp[i][1] == Integer.MAX_VALUE) {
                end = i;
                break;
            }
        }
        int[][] ans = new int[ansTmp.length - (from + 1) - (ansTmp.length - end)][2];
        System.arraycopy(ansTmp, from+1, ans, 0, ans.length);
        return ans;
    }
}
```

#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

题目：

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

注意：解集不能包含重复的组合。



解答：

依旧是模拟题，但是使用DFS来处理。

``` Java
class Solution {

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        for (int i = 1; i <= 50; ++ i) {
            remaining.put(i, 0);
        }
        HashSet<Integer> tmp = new HashSet<>();
        for (int a : candidates) {
            remaining.put(a, remaining.get(a) + 1);
            tmp.add(a);
        }
        nums = new int[tmp.size()];
        int idx = 0;
        for (int a : tmp) {
            nums[idx++] = a;
        }
        f(new LinkedList<>(), target);
        return ans;
    }

    private List<List<Integer>> ans = new ArrayList<>();

    private HashMap<Integer, Integer> remaining = new HashMap<>(50);

    private HashSet<String> filter = new HashSet<>();

    private int[] nums;

    private void f(LinkedList<Integer> stack, int val) {
        for (int num : nums) {
            if (remaining.get(num) > 0) {
                if (num == val) {
                    stack.addLast(num);
                    ff(stack);
                    stack.removeLast();
                } else if (num < val) {
                    remaining.put(num, remaining.get(num) - 1);
                    stack.addLast(num);
                    f(stack, val - num);
                    stack.removeLast();
                    remaining.put(num, remaining.get(num) + 1);
                }
            }
        }
    }

    private void ff(LinkedList<Integer> stack) {
        int[] tmp = new int[stack.size()];
        int idx = 0;
        for (int a : stack) {
            tmp[idx++] = a;
        }
        Arrays.sort(tmp);
        String str = Arrays.toString(tmp);
        if (!filter.contains(str)) {
            ArrayList<Integer> arrayList = new ArrayList<>(stack);
            ans.add(arrayList);
            filter.add(str);
        }
    }
}
```

#### [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

题目：

给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。



解答：

既可以说是背包问题，也可以说是DP问题：

``` Java
class Solution {

    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[1010];
        for (int a : nums) {
            dp[a] = 1;
        }
        for (int i = 1; i <= target; ++ i) {
            for (int j = 0; j < nums.length; ++ j) {
                if (i < nums[j]) {
                    continue;
                }
                dp[i] += dp[i-nums[j]];
            }
        }
        return dp[target];
    }
}
```

#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

题目：

给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] ：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。



解答：

这题依旧是模拟，不过有一个小细节，优化的细节值得我们一提，如果没有这个优化就会超时。因为是四数之和，所以同一个数字最多最多只有四个，即，出现在同一个组合中，第五个相同的数字怎么都不会用到，所以我们提前用HashSet滤一波，可以避免一些重复数据很大的测试用例卡了我们，此外，为了处理负数，我们全部换成正数处理：

``` Java
class Solution {

    public List<List<Integer>> fourSum(int[] nums0, long target) {
        ArrayList<List<Integer>> ans = new ArrayList<>();
        HashSet<String> marked = new HashSet<>();
        HashMap<Integer, Integer> set = new HashMap<>();
        for (int a : nums0) {
            Integer val;
            if ((val = set.get(a)) == null) {
                set.put(a, 1);
            } else {
                if (val < 4) {
                    set.put(a, val + 1);
                }
            }
        }
        long[] nums = new long[nums0.length];
        long upperBound = 1000000001;
        int len = 0;
        for (Map.Entry<Integer, Integer> a : set.entrySet()) {
            for (int i = 0; i < a.getValue(); ++ i) {
                nums[len++] = a.getKey() + upperBound;
            }
        }
        target += 4L * upperBound;
        Arrays.sort(nums, 0, len);
        for (int i = 0; i < len; ++i) {
            if (nums[i] > target) {
                return ans;
            }
            for (int j = i + 1; j < len; ++j) {
                if (nums[i] + nums[j] > target) {
                    return ans;
                }
                for (int k = j + 1; k < len; ++k) {
                    if (nums[i] + nums[j] + nums[k] > target) {
                        return ans;
                    }
                    int idx = Arrays.binarySearch(nums, k + 1, len, target - nums[i] - nums[j] - nums[k]);
                    if (idx < 0) {
                        continue;
                    } else {
                        ArrayList<Integer> list = new ArrayList<>(4);
                        list.add((int) (nums[i] - upperBound));
                        list.add((int) (nums[j] - upperBound));
                        list.add((int) (nums[k] - upperBound));
                        list.add((int) (nums[idx] - upperBound));
                        String str = list.toString();
                        if (!marked.contains(str)) {
                            ans.add(list);
                            marked.add(str);
                        }
                    }
                }
            }
        }
        return ans;
    }
}
```

#### [1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

题目：

给出一个字符串 s（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中 不应 包含任何括号。



解答：

使用栈模拟即可。

``` Java
class Solution {
    public String reverseParentheses(String s) {
        Stack<StringBuilder> stack = new Stack<>();
        stack.push(new StringBuilder());
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                StringBuilder buffer = new StringBuilder();
                stack.push(buffer);
            } else if (s.charAt(i) == ')') {
                StringBuilder pop = stack.pop();
                StringBuilder reverse = pop.reverse();
                stack.peek().append(reverse);
            } else {
                stack.peek().append(s.charAt(i));
            }
        }
        return stack.peek().toString();
    }
}
```

