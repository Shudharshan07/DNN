

build:
	@npm --prefix ./frontend run build -- --outDir ../services/websocket/dist
	@GOOS=windows go -C services/websocket build -ldflags="-H=windowsgui" -o ../../bin/app.exe .
	@bin/app.exe

win:
	@bin/app.exe
