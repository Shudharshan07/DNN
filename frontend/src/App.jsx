import { useEffect } from "react";
import socket from "./api/data";

function App() {
  useEffect(() => {
    socket.onopen = () => {
      console.log("Connected");
    };

    socket.onmessage = (event) => {
      console.log("Message:", event.data);
    };
  }, []);

  return <>
  hiii
  </>
}

export default App