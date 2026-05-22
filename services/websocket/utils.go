package main

import "fmt"

func SuccessText(format string, a ...any) string {
	return fmt.Sprintf("\x1b[42m\x1b[30m SUCCESS \x1b[0m   "+format+"\n", a...)
}

func FailedText(format string, a ...any) string {
	return fmt.Sprintf("\x1b[41m\x1b[37m  FAILED \x1b[0m   "+format+"\n", a...)
}

// get the port from env
