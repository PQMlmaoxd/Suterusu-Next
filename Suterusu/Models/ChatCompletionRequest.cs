using System.Collections.Generic;
using Newtonsoft.Json;

namespace Suterusu.Models
{
    public class ChatCompletionRequest
    {
        public string Model { get; set; }

        public List<ChatRequestMessage> Messages { get; set; }

        public double Temperature { get; set; } = 0.7;

        public int? MaxTokens { get; set; }

        public string ReasoningEffort { get; set; }
    }

    public class OllamaChatRequest
    {
        public string Model { get; set; }

        public List<OllamaChatMessage> Messages { get; set; }

        public bool Stream { get; set; }

        public bool Think { get; set; }

        public OllamaChatOptions Options { get; set; }
    }

    public class OllamaChatMessage
    {
        public string Role { get; set; }

        public string Content { get; set; }

        public List<string> Images { get; set; }
    }

    public class OllamaChatOptions
    {
        public double? Temperature { get; set; }
    }
}
