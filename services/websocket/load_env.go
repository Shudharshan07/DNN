package main

import (
	"bufio"
	"os"
	"strings"
)

func LoadEnv() (int, error) {
	file, err := os.Open(".env")
	if err != nil {
		return 0, err
	}
	defer file.Close()

	s := bufio.NewScanner(file)
	c := 0

	for s.Scan() {
		line := s.Text()
		if len(line) < 3 || line[0] == '#' {
			continue
		}

		vals := strings.SplitN(line, "=", 2)

		if len(vals) == 2 {
			err := os.Setenv(vals[0], vals[1])

			if err == nil {
				c++
			}
		}
	}

	return c, nil
}
