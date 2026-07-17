package main

import (
	"DNN/services/reader"
	"context"
	"fmt"
	"os"
	"time"
)

func SuccessText(format string, a ...any) string {
	return fmt.Sprintf("\x1b[42m\x1b[30m SUCCESS \x1b[0m   "+format+"\n", a...)
}

func FailedText(format string, a ...any) string {
	return fmt.Sprintf("\x1b[41m\x1b[37m  FAILED \x1b[0m   "+format+"\n", a...)
}

// get the port from env
func StartStream(shmName string, pollInterval time.Duration, stop context.Context, out chan *reader.Snapshot) {
	reader := reader.NewSharedMemoryReader(shmName)

	if err := reader.Open(); err != nil {
		fmt.Print(FailedText("error: %v\n", err))
		fmt.Print(FailedText("Make sure the C++ process is running and using shm_name=%q\n", shmName))
		reader.Close()
		os.Exit(1)
	}
	defer reader.Close()

	reader.Run(pollInterval, stop, out)
}
