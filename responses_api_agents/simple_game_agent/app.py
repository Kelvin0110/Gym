import json
from typing import Dict, Any

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.server_utils import ResourcesServerRef, ModelServerRef

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class SimpleGameAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_moves: int = 50  # Maximum number of moves allowed


class SimpleGameAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    # Game-specific parameters will be passed through to get_initial_board
    clues: int = 30
    scale: int = 9


class SimpleGameAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleGameAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleGameAgent(SimpleResponsesAPIAgent):
    config: SimpleGameAgentConfig
    
    # Add a class attribute to temporarily store game params
    _current_game_params: dict = {}

    async def responses(
        self, 
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
        game_params: dict = None
    ) -> NeMoGymResponse:
        """Run the game with direct model-environment communication - no tools needed."""
        
        if game_params is None:
            game_params = {}
            
        conversation = body["input"].copy()
        moves_made = 0
        game_state = None
        reward = 0.0
        is_complete = False
        
        # NEW: Accumulate model outputs like simple_agent does
        new_outputs = []
        
        # Step 1: Get initial board from environment
        try:
            game_params = self._current_game_params
            
            initial_board_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/get_initial_board",
                json=game_params,
            )
            board_data = initial_board_response.json()
            game_state = board_data.get("game_state")
            
            # Add the board and instructions to conversation
            game_intro = {
                "role": "assistant",
                "content": f"{board_data['board_text']}\n\n{board_data['instructions']}"
            }
            conversation.append(game_intro)
            
        except Exception as e:
            error_msg = {
                "role": "assistant", 
                "content": f"Error initializing game: {str(e)}"
            }
            conversation.append(error_msg)
            return {
                "output": [{"type": "text", "text": error_msg["content"]}]
            }

        # Step 2: Game loop continues...
        while moves_made < self.config.max_moves:
            # Get model response
            model_body = NeMoGymResponseCreateParamsNonStreaming(
                input=conversation,
                tools=[],
            )
            
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=model_body,
            )
            model_response = NeMoGymResponse.model_validate(model_response.json())
            
            print("== MODEL RESPONSE ==")
            print(model_response)
            print("== MODEL RESPONSE ==")
            
            # NEW: Accumulate model outputs like simple_agent does
            output = model_response.output
            new_outputs.extend((o.model_dump() for o in output))
            
            # Extract the text response
            output_item = model_response.output[-1]

            # 1️⃣ accept both plain text-style or message style
            if output_item.type not in {"message", "text"}:
                break

            # 2️⃣ pull the text out correctly
            if output_item.type == "message":
                # concatenate every output_text chunk
                model_text = "".join(
                    part.text for part in output_item.content
                    if part.type == "output_text"
                )
            else:   # legacy 'text'
                model_text = output_item.text
            conversation.append({"role": "assistant", "content": model_text})
            
            # Pass model's raw text directly to environment
            try:
                move_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/make_move",
                    json={
                        "game_state": game_state,
                        "move": model_text
                    },
                )
                move_data = move_response.json()
                game_state = move_data.get("game_state")
                moves_made += 1
                reward += move_data.get("move_reward", 0.0)
                
                # Add environment's response back to conversation
                env_feedback = {
                    "role": "user",
                    "content": f"{move_data['message']}\n\n{move_data['board_text']}"
                }
                conversation.append(env_feedback)
                
                print("== ENVIRONMENT RESPONSE ==")
                print(move_data)
                print("== ENVIRONMENT RESPONSE ==")
                
                # Check if game is complete
                if move_data.get("is_complete", False):
                    is_complete = True
                    completion_msg = {
                        "role": "user",
                        "content": "🎉 Game completed successfully!"
                    }
                    conversation.append(completion_msg)
                    break
                    
            except Exception as e:
                error_msg = {
                    "role": "user",
                    "content": f"Move error: {str(e)}"
                }
                conversation.append(error_msg)

        # Store metrics for verify step
        self._reward = reward
        self._total_moves = moves_made
        self._is_complete = is_complete
        # NEW: Store conversation for later use
        # self._final_conversation = conversation

        # NEW: Return accumulated outputs like simple_agent does
        final_response_dict = model_response.model_dump()
        final_response_dict["output"] = new_outputs
        return final_response_dict
    
    def _format_conversation(self, conversation: list) -> str:
        """Format the conversation into a readable string."""
        formatted_parts = []
        for msg in conversation:
            role = msg["role"].upper()
            content = msg["content"]
            formatted_parts.append(f"[{role}]: {content}")
        return "\n\n".join(formatted_parts)

    async def run(self, body: SimpleGameAgentRunRequest) -> SimpleGameAgentVerifyResponse:
        """Run a complete game session."""
        
        # Prepare the conversation
        conversation_body = NeMoGymResponseCreateParamsNonStreaming(
            input=body.responses_create_params["input"],
            tools=[],
        )
        
        # Extract game parameters
        game_params = {k: v for k, v in body.model_dump().items() 
                      if k not in ["responses_create_params"]}

        # Store in class attribute instead of trying to setattr on the TypedDict
        self._current_game_params = game_params

        # Run the game with game_params passed directly
        response = await self.responses(conversation_body, game_params)

        # Create verify request  
        verify_request = SimpleGameAgentVerifyRequest.model_validate(
            body.model_dump() | {
                "response":      response,                 # OpenAI response
                "reward":  self._reward,       # numbers we stored
                "total_moves":   self._total_moves,
                "is_complete":   self._is_complete,
            }
        )
        
        # Call verify on the resources server
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        
        return SimpleGameAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    SimpleGameAgent.run_webserver()