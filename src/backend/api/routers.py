from utils.callbacks import StreamingLLMCallbackHandler
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from integrations.openai import get_openai_chain
from schemas.message import ChatResponse

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_hanlder = StreamingLLMCallbackHandler(websocket)
    conversation_chain = get_openai_chain(stream_hanlder)
    while True:
        try:
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
            await websocket.close()
            break
        except Exception as e:
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
