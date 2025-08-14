import json
from typing import Dict, Any

from pydantic import ConfigDict

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput

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


class TextGameAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_moves: int = 50  # Maximum number of moves allowed


class TextGameAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    # Game-specific parameters
    clues: int = 30
    scale: int = 9


class TextGameAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class TextGameAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class TextGameAgent(SimpleResponsesAPIAgent):
    config: TextGameAgentConfig

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Run the text game with iterative model calls."""
        new_outputs = []
        game_state = None
        moves_made = 0
        
        while moves_made < self.config.max_moves:
            # Prepare the current conversation state
            new_body: NeMoGymResponseCreateParamsNonStreaming = body.copy()
            new_body["input"] = body["input"] + new_outputs

            # Get model response
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
            )
            model_response = NeMoGymResponse.model_validate(model_response.json())

            output = model_response.output
            new_outputs.extend((o.model_dump() for o in output))
            
            # Check if model wants to make a function call
            if output[-1].type != "function_call":
                # Model provided a regular response, end the game
                break

            output_function_call: ResponseFunctionToolCall = output[-1]

            # Handle different function calls
            if output_function_call.name in ["get_initial_board", "make_move"]:
                # Call the sudoku resources server
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                )

                # Create function call output
                tool_response = FunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=json.dumps(api_response.json()),
                )
                new_outputs.append(tool_response)

                # Track game state and moves
                response_data = api_response.json()
                if "game_state" in response_data:
                    game_state = response_data["game_state"]
                
                if output_function_call.name == "make_move":
                    moves_made += 1
                    
                    # Check if game is complete
                    if response_data.get("is_complete", False):
                        # Game completed successfully!
                        completion_message = {
                            "type": "text",
                            "text": "Congratulations! You have successfully completed the Sudoku puzzle!"
                        }
                        new_outputs.append(completion_message)
                        break
            else:
                # Unknown function call - return error
                error_response = FunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=json.dumps({
                        "error": f"Unknown function: {output_function_call.name}",
                        "available_functions": ["get_initial_board", "make_move"]
                    }),
                )
                new_outputs.append(error_response)

        # Prepare final response
        final_response_dict = model_response.model_dump()
        final_response_dict["output"] = new_outputs
        return final_response_dict

    async def run(self, body: TextGameAgentRunRequest) -> TextGameAgentVerifyResponse:
        """Run a complete game session."""
        
        # Prepare the initial game setup
        initial_prompt = {
            "role": "system",
            "content": (
                "You are playing Sudoku! Start by calling get_initial_board to get the puzzle, "
                "then use make_move to place numbers. Follow the instructions carefully."
            )
        }
        
        user_start = {
            "role": "user", 
            "content": f"Let's play Sudoku! Please start a new game with {body.clues} clues on a {body.scale}x{body.scale} board."
        }

        # Set up the tools/functions available to the model
        tools = [
            {
                "type": "function",
                "name": "get_initial_board",
                "description": "Get the initial Sudoku board and instructions to start the game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clues": {
                            "type": "integer",
                            "description": "Number of pre-filled cells",
                            "default": body.clues
                        },
                        "scale": {
                            "type": "integer", 
                            "description": "Size of the Sudoku grid (4 or 9)",
                            "default": body.scale
                        }
                    },
                    "required": ["clues", "scale"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "make_move",
                "description": "Make a move in the Sudoku game by placing a number in a cell",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_state": {
                            "type": "object",
                            "description": "Current game state from previous function calls",
                            "additionalProperties": False,
                        },
                        "move": {
                            "type": "string",
                            "description": "Your move in the format \\boxed{row column number}, e.g. \\boxed{1 1 5}"
                        }
                    },
                    "required": ["game_state", "move"],
                    "additionalProperties": False,
                },
                "strict": False,
            }
        ]

        # Create the responses body
        responses_body = NeMoGymResponseCreateParamsNonStreaming(
            input=[initial_prompt, user_start],
            tools=tools,
        )

        # Run the game
        response = await self.responses(responses_body)

        # Create verify request  
        verify_request = TextGameAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response}
        )
        
        # Call verify on the resources server
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        
        return TextGameAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    TextGameAgent.run_webserver()