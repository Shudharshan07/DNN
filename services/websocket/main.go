package main

import (
	"DNN/services/reader"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

var out = make(chan *reader.Snapshot)

func handler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Print(FailedText(err.Error()))
	}

	WriteLoop(conn, out)
}

func WriteLoop(conn *websocket.Conn, out chan *reader.Snapshot) {

	for snap := range out {
		conn.WriteJSON(snap)
	}
}

func main() {
	c, err := LoadEnv()
	if err != nil {
		fmt.Print(FailedText(err.Error()))
	} else {
		fmt.Print(SuccessText("Loaded %d env variables", c))
	}

	port := os.Getenv("PORT")
	SHM_NAME := os.Getenv("SHM_NAME")

	http.HandleFunc("/ws", handler)

	server := &http.Server{
		Addr:    ":" + port,
		Handler: nil,
	}

	go func() {
		fmt.Print(SuccessText("Serving on %s", port))
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Print(FailedText(err.Error()))
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go StartStream(SHM_NAME, time.Second, ctx, out)

	<-ctx.Done()
	server.Shutdown(ctx)

	shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	server.Shutdown(shutCtx)
}
