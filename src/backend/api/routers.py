from utils.callbacks import StreamingLLMCallbackHandler
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from integrations.openai import get_openai_chain, settings
from schemas.message import ChatResponse

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    if not settings.OPENAI_API_KEY.startswith("sk-"):
            await websocket.send_json({"error": "OPENAI_API_KEY is not set"})
            return

    stream_hanlder = StreamingLLMCallbackHandler(websocket)
    conversation_chain = get_openai_chain(stream_hanlder)
    try:
        while True:
             # Receive and send back the client message
            user_msg = await websocket.receive_text()
            resp = ChatResponse(sender="human", message=user_msg, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # Send the message to the chain and feed the response back to the client
            output = await conversation_chain.acall(
                {
                    "input": user_msg,
                }
            )

            # Send the end-response back to the client
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        resp = ChatResponse(
            sender="bot",
            message="Sorry, something went wrong. Try again.",
            type="error",
        )
        await websocket.send_json(resp.dict())
