package main

import (
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

func handler(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Print(err)
	}

	WriteLoop(conn)
}

func WriteLoop(conn *websocket.Conn) {

	for {
		msg := time.Now().String()
		conn.WriteMessage(1, []byte(msg))
		time.Sleep(1 * time.Second)
	}
}

func main() {
	http.HandleFunc("/ws", handler)

	server := &http.Server{
		Addr:    ":9000",
		Handler: nil,
	}

	go func() {
		fmt.Print("Serving")
		// fmt.Println("\x1b[37;42mServing on :9000\x1b[0m")
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGINT)
	defer stop()

	<-ctx.Done()
	server.Shutdown(ctx)
}
