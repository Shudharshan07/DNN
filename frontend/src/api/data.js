

const url = "ws://localhost:9000/ws"
const socket = new WebSocket(url)

socket.onopen = () => {
    console.log(`Connected to ${url}`)
}

socket.onmessage = (event) => {
    console.log(event)
}

export default socket
