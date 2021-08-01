package airbnb

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// P251
// Vector2D - implements an iterator over a 2d vector
type Vector2D struct {
	row    int
	col    int
	values [][]int
}

func Constructor(vec [][]int) Vector2D {
	return Vector2D{values: vec}
}

func (vec *Vector2D) Next() int {
	if vec.col == len(vec.values[vec.row]) {
		vec.row, vec.col = vec.row+1, 0
		return vec.Next()
	}
	res := vec.values[vec.row][vec.col]
	vec.col++
	return res
}

func (vec *Vector2D) HasNext() bool {
	return vec.hasNextHelper(vec.row, vec.col)
}

func (vec *Vector2D) hasNextHelper(row int, col int) bool {
	if row < len(vec.values) {
		if col == len(vec.values[row]) {
			row, col = row+1, 0
			return vec.hasNextHelper(row, col)
		} else {
			return true
		}
	}
	return false
}

// P1387 - Sort integers by power value
func getKth(lo int, hi int, k int) int {
	// generate the array
	nums := [][]int{}
	memo := map[int]int{1: 0}
	i, p := lo, 0
	for i <= hi {
		p = power(i, memo)
		nums = append(nums, []int{i, p})
		i++
	}
	// sort by power value
	sort.SliceStable(nums, func(i, j int) bool {
		return (nums[i][1] < nums[j][1]) || (nums[i][0] < nums[j][0])
	})
	return nums[k-1][0]
}

func power(x int, memo map[int]int) int {
	p := 0
	if val, ok := memo[x]; ok {
		return val
	}
	if x%2 == 0 {
		p = 1 + power(x/2, memo)
	} else {
		p = 1 + power(3*x+1, memo)
	}
	memo[x] = p
	return p
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// P297 - Serialize/Deserialize BT
func serialize(root *TreeNode) string {
	res := dfsSerialize(root)
	fmt.Println(res)
	return strings.Join(res, ",")
}

func dfsSerialize(root *TreeNode) []string {
	if root == nil {
		return []string{"x"}
	}
	res, left, right := []string{}, []string{}, []string{}
	res = append(res, strconv.Itoa(root.Val))
	left = dfsSerialize(root.Left)
	res = append(res, left...)
	right = dfsSerialize(root.Right)
	res = append(res, right...)
	return res
}

// Deserializes your encoded data to tree.
func deserialize(data string) *TreeNode {
	nodes := strings.Split(data, ",")
	root, _ := dfsDeserialize(nodes, 0)
	return root
}

func dfsDeserialize(nodes []string, curr int) (*TreeNode, int) {
	if curr == len(nodes) {
		return nil, curr
	}
	s := nodes[curr]
	curr++
	if s == "x" {
		return nil, curr
	}
	var root, left, right *TreeNode
	val, _ := strconv.Atoi(s)
	root = &TreeNode{Val: val}
	left, curr = dfsDeserialize(nodes, curr)
	right, curr = dfsDeserialize(nodes, curr)
	if left != nil {
		root.Left = left
	}
	if right != nil {
		root.Right = right
	}

	return root, curr
}

//P236 - LCA of BT (binary tree)
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// if one of the values is the root node
	if root == nil || root == q || root == p {
		return root
	}
	// recurse on left side
	var left, right *TreeNode
	if root.Left != nil {
		left = lowestCommonAncestor(root.Left, p, q)
	}
	if root.Right != nil {
		right = lowestCommonAncestor(root.Right, p, q)
	}
	if left != nil && right != nil {
		return root
	} else if left != nil {
		return left
	} else {
		return right
	}
}

//P322 - Coin Change (DP)
func coinChange(coins []int, amount int) int {
	change := make([]int, amount+1)
	change[0] = 0
	for i := 1; i < len(change); i++ {
		change[i] = math.MaxInt32
		for _, c := range coins {
			if c > i {
				continue
			}
			change[i] = min(change[i-c]+1, change[i])
		}
	}
	if change[amount] == math.MaxInt32 {
		return -1
	}
	return change[amount]
}

//P166 - Fraction to recurring decimal
func fractionToDecimal(numerator int, denominator int) string {
	if numerator == 0 {
		return "0"
	}
	if denominator == 0 {
		return ""
	}
	// flag for positive/negative
	flag := 1
	if numerator < 0 {
		flag *= -1
		numerator *= -1
	}
	if denominator < 0 {
		flag *= -1
		denominator *= -1
	}
	var div, rem strings.Builder
	var final string
	dividend, remainder := 0, 0
	// deal with part before the decimal point
	dividend, remainder = numerator/denominator, numerator%denominator
	div.WriteString(strconv.Itoa(dividend))
	if remainder > 0 {
		div.WriteString(".")
	}
	// deal with remainder (part after decimal point)
	memo := make(map[int]int)
	var pos int
	for remainder > 0 {
		numerator = remainder * 10
		dividend = numerator / denominator
		if idx, ok := memo[remainder]; ok && dividend > 0 {
			rem.WriteString(")")
			final = rem.String()
			final = final[:idx] + "(" + final[idx:]
			break
		}
		memo[remainder] = pos
		pos++
		remainder = numerator % denominator
		rem.WriteString(strconv.Itoa(dividend))
	}
	if len(final) == 0 {
		final = rem.String()
	}

	if flag < 0 {
		return "-" + div.String() + final
	}
	return div.String() + final

}

//P 336 - Palindrome Pairs
func palindromePairs(words []string) [][]int {
	res := [][]int{}
	//1. construct word map for easier lookup
	wordMap := constructWordMap(words)
	var rev, s string
	//2. checking for palindromes
	//a. if reverse of word is present in map
	//b. for all suffixes, see if their reverses present
	//c. for all prefixes, see if their reverses present
	for idx, w := range words {
		// case a ( only for words greater than length 1)
		if len(w) > 1 {
			rev = reverse(w)
			if j, ok := wordMap[rev]; ok {
				res = append(res, []int{idx, j})
			}
		}
		// case b - suffixes (word 1 is combining with a shorter word)
		for i := 1; i < len(w); i++ {
			s, rev = w[i:], reverse(w[i:])
			// if suffix is a palindrome, then if reverse of prefix is present in word map, then we can form a palindrome
			if rev == s {
				if j, ok := wordMap[reverse(w[:i])]; ok {
					res = append(res, []int{idx, j})
				}
			}
		}
		// case c: deal with the prefixes (word 1 is combining with smaller words)
		for i := 1; i < len(w); i++ {
			s, rev = w[:i], reverse(w[:i])
			if rev == s {
				if j, ok := wordMap[reverse(w[i:])]; ok {
					res = append(res, []int{j, idx})
				}
			}
		}
	}
	return res
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func constructWordMap(words []string) map[string]int {
	wordMap := map[string]int{}
	for idx, w := range words {
		wordMap[w] = idx
	}
	return wordMap
}

//P755 - Pour water
func pourWater(heights []int, volume int, k int) []int {
	var i, flowIdx int
	diff := []int{-1, 1}
	for volume > 0 {
		flowIdx = k
		for _, d := range diff {
			i = k + d
			for i >= 0 && i < len(heights) && heights[i] <= heights[i-d] {
				if heights[flowIdx] > heights[i] {
					flowIdx = i
				}
				i += d
			}
			if flowIdx != k {
				break
			}
		}
		heights[flowIdx] += 1
		volume -= 1
	}
	return heights
}

type FileSystem struct {
	name     string
	isDir    bool
	children map[string]*FileSystem
	content  string
}

func NewFileSystem() FileSystem {
	return FileSystem{name: "/", isDir: true}
}

func (fs *FileSystem) Ls(path string) []string {
	if path == fs.name {
		return fs.getDirectoryContents()
	}
	pathArr := strings.Split(path, "/")
	// remove root
	if pathArr[0] == "" {
		pathArr = pathArr[1:]
	}
	for _, p := range pathArr {
		if child, ok := fs.children[p]; ok {
			fs = child
		} else {
			return []string{}
		}
	}
	return fs.getDirectoryContents()

}

func (fs *FileSystem) getDirectoryContents() []string {
	res := []string{}
	for name, _ := range fs.children {
		res = append(res, name)
	}
	return res
}

func (fs *FileSystem) Mkdir(path string) {
	pathArr := strings.Split(path, "/")
	// remove root
	if pathArr[0] == "" {
		pathArr = pathArr[1:]
	}
	for _, p := range pathArr {
		if _, ok := fs.children[p]; !ok {
			fs.children[p] = &FileSystem{name: p, isDir: true}
		}
		fs = fs.children[p]
	}
}

func (fs *FileSystem) AddContentToFile(filePath string, content string) {
}

func (fs *FileSystem) ReadContentFromFile(filePath string) string {
	return ""
}
