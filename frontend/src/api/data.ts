const url = `${import.meta.env.NEXT_PUBLIC_WS_URL}/ws`
const socket = new WebSocket(url)

socket.onopen = () => {
    console.log(`Connected to ${url}`)
}

socket.onmessage = (event) => {
    console.log(event)
}

export default socket
