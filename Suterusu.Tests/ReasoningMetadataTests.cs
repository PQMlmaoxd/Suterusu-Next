using System.Linq;
using Newtonsoft.Json.Linq;
using Suterusu.Models;
using Suterusu.UI;
using Xunit;

namespace Suterusu.Tests
{
    public class ReasoningMetadataTests
    {
        [Fact]
        public void ExtractReasoningEfforts_ReadsCommonDirectMetadataFields()
        {
            var metadata = JObject.Parse(@"{
                ""reasoning_efforts"": [""none"", ""low""],
                ""supported_reasoning_levels"": [""medium"", ""high""],
                ""reasoning"": { ""levels"": [""xhigh""] },
                ""capabilities"": { ""reasoning_efforts"": [""low"", ""custom-level""] }
            }");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata).ToList();

            Assert.Equal(new[] { "none", "low", "medium", "high", "xhigh", "custom-level" }, efforts);
        }

        [Fact]
        public void ExtractReasoningEfforts_DoesNotInferFromModelName()
        {
            var metadata = JObject.Parse(@"{ ""id"": ""gpt-5.5"" }");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata);

            Assert.Empty(efforts);
        }

        [Fact]
        public void ExtractReasoningEfforts_DoesNotInventLevelsFromSupportedParameters()
        {
            var metadata = JObject.Parse(@"{
                ""id"": ""openrouter-model"",
                ""supported_parameters"": [""temperature"", ""reasoning"", ""include_reasoning""]
            }");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata).ToList();

            Assert.Empty(efforts);
        }

        [Fact]
        public void ExtractReasoningEfforts_ReadsNestedReasoningEffortFields()
        {
            var metadata = JObject.Parse(@"{
                ""reasoning"": { ""efforts"": [""minimal"", ""high""] },
                ""capabilities"": { ""reasoning"": { ""levels"": [""xhigh""] } }
            }");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata).ToList();

            Assert.Equal(new[] { "minimal", "high", "xhigh" }, efforts);
        }

        [Fact]
        public void ExtractReasoningEfforts_IgnoresArrayShapedMetadata()
        {
            var metadata = JObject.Parse(@"{
                ""id"": ""odd-model"",
                ""reasoning"": [],
                ""capabilities"": []
            }");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata);

            Assert.Empty(efforts);
        }

        [Fact]
        public void ExtractReasoningEfforts_IgnoresArrayRootMetadata()
        {
            var metadata = JArray.Parse(@"[""odd-model""]");

            var efforts = ModelPriorityEditor.ExtractReasoningEfforts(metadata);

            Assert.Empty(efforts);
        }

        [Fact]
        public void ExtractReasoningEffortsFromDetails_ReadsEndpointReasoningLevels()
        {
            var details = JObject.Parse(@"{
                ""data"": {
                    ""endpoints"": [
                        { ""reasoning"": { ""levels"": [""low"", ""high""] } },
                        { ""capabilities"": { ""reasoning"": { ""efforts"": [""xhigh""] } } }
                    ]
                }
            }");

            var efforts = ModelPriorityEditor.ExtractReasoningEffortsFromDetails(details).ToList();

            Assert.Equal(new[] { "low", "high", "xhigh" }, efforts);
        }

        [Fact]
        public void BuildModelsUrl_UsesOllamaNativeTagsEndpoint()
        {
            Assert.Equal(
                "http://localhost:11434/api/tags",
                ModelPriorityEditor.BuildModelsUrl("http://localhost:11434/api/chat"));
        }

        [Theory]
        [InlineData("https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/models")]
        [InlineData("https://openrouter.ai/api/v1/chat/completions", "https://openrouter.ai/api/v1/models")]
        [InlineData("http://localhost:8080/v1/chat/completions", "http://localhost:8080/v1/models")]
        [InlineData("http://127.0.0.1:8317/v1", "http://127.0.0.1:8317/v1/models")]
        [InlineData("https://example.test/v1", "https://example.test/v1/models")]
        public void BuildModelsUrl_PreservesOpenAiCompatiblePresetRouting(string baseUrl, string expected)
        {
            Assert.Equal(expected, ModelPriorityEditor.BuildModelsUrl(baseUrl));
        }

        [Fact]
        public void GetModelId_ReadsOllamaTagsName()
        {
            var metadata = JObject.Parse(@"{ ""name"": ""llama3.2:latest"", ""model"": ""llama3.2:latest"" }");

            Assert.Equal("llama3.2:latest", ModelPriorityEditor.GetModelId(metadata));
        }

        [Fact]
        public void GetModelId_ReadsStringModelEntry()
        {
            Assert.Equal("llama3.2:latest", ModelPriorityEditor.GetModelId(new JValue("llama3.2:latest")));
        }

        [Fact]
        public void OllamaPreset_UsesNativeApiChatEndpoint()
        {
            var preset = EndpointPreset.FindByName("Ollama");

            Assert.Equal("http://localhost:11434/api/chat", preset.BaseUrl);
            Assert.False(preset.RequiresApiKey);
        }
    }
}
