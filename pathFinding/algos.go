package pathfinding

import (
	"encoding/json"
	"fmt"
)

//P 773: Sliding Puzzle
type BoardState struct {
	board [][]int
	bRow  int
	bCol  int
	moves int
}

func slidingPuzzle(board [][]int) int {
	return bfsSearch(board, findBlankTile(board))
}

func bfsSearch(state [][]int, blankPos []int) int {
	fmt.Println("Board State ", state)
	var curr, tmp [][]int
	var boardState BoardState
	var moves, bRow, bCol, nRow, nCol int
	var boardStr string
	seen := map[string]bool{} // to store states we have already processed
	// init Q
	Q := []BoardState{BoardState{board: state, bRow: blankPos[0], bCol: blankPos[1], moves: 0}}
	for len(Q) > 0 {
		curr, bRow, bCol, moves, Q = Q[0].board, Q[0].bRow, Q[0].bCol, Q[0].moves, Q[1:]
		if isSolved(curr) {
			return moves
		}
		boardStr = boardToString(curr)
		if _, ok := seen[boardStr]; ok {
			continue
		}
		seen[boardStr] = true
		// process neighbors
		for _, n := range getNeighbors(curr, []int{bRow, bCol}) {
			nRow, nCol = n[0], n[1]
			tmp = copy2dSlice(curr)
			tmp[nRow][nCol], tmp[bRow][bCol] = tmp[bRow][bCol], tmp[nRow][nCol]
			boardState = BoardState{board: tmp, bRow: nRow, bCol: nCol, moves: moves + 1}
			Q = append(Q, boardState)
		}
	}
	return -1
}

func copy2dSlice(state [][]int) [][]int {
	tmp := make([][]int, len(state))
	for i, _ := range state {
		tmp[i] = make([]int, len(state[i]))
		copy(tmp[i], state[i])
	}
	return tmp
}

func findBlankTile(board [][]int) []int {
	for i := range board {
		for j := range board[0] {
			if board[i][j] == 0 {
				return []int{i, j}
			}
		}
	}
	return []int{-1, -1}
}

func getNeighbors(state [][]int, blank []int) [][]int {
	row, col := blank[0], blank[1]
	var i, j int
	neighbors := [][]int{}
	deltaRow, deltaCol := []int{0, -1, 0, 1}, []int{-1, 0, 1, 0}
	for idx, _ := range deltaRow {
		i, j = row+deltaRow[idx], col+deltaCol[idx]
		if i >= 0 && i < len(state) && j >= 0 && j < len(state[0]) {
			neighbors = append(neighbors, []int{i, j})
		}
	}
	return neighbors
}

func boardToString(board [][]int) string {
	j, _ := json.Marshal(board)
	return string(j)
}

func isSolved(state [][]int) bool {
	solved := [][]int{[]int{1, 2, 3}, []int{4, 5, 0}}
	return boardToString(solved) == boardToString(state)
}

//P 200 - number of islands ( finding components in an undirected graph)
func numIslands(grid [][]byte) int {
	numIslands, seen, Q := 0, make([][]bool, len(grid)), [][]int{}
	// init seen array
	for i := range seen {
		seen[i] = make([]bool, len(grid[0]))
	}
	var node byte
	var point []int
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			node = grid[i][j]
			if seen[i][j] || node == '0' {
				continue
			}
			Q = append(Q, []int{i, j})
			for len(Q) > 0 {
				point, Q = Q[0], Q[1:]
				// check if we have already seen this position
				if seen[point[0]][point[1]] {
					continue
				}
				seen[point[0]][point[1]] = true
				for _, n := range getNeighboringIslands(grid, point) {
					Q = append(Q, n)
				}
			}
			numIslands++
		}
	}
	return numIslands
}

func getNeighboringIslands(grid [][]byte, point []int) [][]int {
	neighbors := [][]int{}
	rowDelta, colDelta := []int{0, -1, 0, 1}, []int{-1, 0, 1, 0}
	row, col := point[0], point[1]
	var nRow, nCol int
	for i := range rowDelta {
		nRow, nCol = row+rowDelta[i], col+colDelta[i]
		if nRow == len(grid) || nRow < 0 || nCol == len(grid[0]) || nCol < 0 || grid[nRow][nCol] == '0' {
			continue
		}
		neighbors = append(neighbors, []int{nRow, nCol})
	}
	return neighbors
}

//P 62 - find unique paths in a grid recursively ( inefficient as this is done without caching )
func uniquePathsRecursive(row int, col int) int {
	seen := make([][]bool, row)
	// init rest of seen
	for i := range seen {
		seen[i] = make([]bool, col)
	}
	var currRow, currCol int
	return dfsTraversal(row, col, currRow, currCol, seen)
}

func dfsTraversal(row int, col int, currRow int, currCol int, seen [][]bool) int {
	var paths int
	if currRow == row-1 && currCol == col-1 {
		return 1 + paths
	}
	var nRow, nCol int
	for _, n := range getNeighbors(row, col, currRow, currCol) {
		nRow, nCol = n[0], n[1]
		if seen[nRow][nCol] {
			continue
		}
		seen[nRow][nCol] = true
		paths += dfsTraversal(row, col, nRow, nCol, seen)
		seen[nRow][nCol] = false
	}
	return paths
}

func getNeighbors(row int, col int, currRow int, currCol int) [][]int {
	rowDelta, colDelta := []int{1, 0}, []int{0, 1}
	var nRow, nCol int
	var neighbors [][]int
	for i := range rowDelta {
		nRow, nCol = currRow+rowDelta[i], currCol+colDelta[i]
		if nRow >= 0 && nRow < row && nCol >= 0 && nCol < col {
			neighbors = append(neighbors, []int{nRow, nCol})
		}
	}
	return neighbors
}

// P 62 - Unique paths I via DP (only right and bottom moves allowed)
func uniquePaths(row int, col int) int {
	prev, curr := make([]int, col), make([]int, col)
	for i := 0; i < col; i++ {
		prev[i], curr[i] = 1, 1
	}
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			curr[j] = curr[j-1] + prev[j]
		}
		curr, prev = prev, curr
	}
	return prev[col-1]
}
