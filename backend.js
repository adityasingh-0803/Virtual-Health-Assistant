mkdir server
cd server
npm init -y
npm install express mongoose body-parser axios dotenv cors

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { Configuration, OpenAIApi } = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

// OpenAI API Configuration
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// API Route for health queries
app.post('/api/ask', async (req, res) => {
  const { query } = req.body;

  try {
    const response = await openai.createCompletion({
      model: 'text-davinci-003',
      prompt: query,
      max_tokens: 150,
    });

    res.json({ answer: response.data.choices[0].text.trim() });
  } catch (error) {
    console.error(error);
    res.status(500).send('Error processing request');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});