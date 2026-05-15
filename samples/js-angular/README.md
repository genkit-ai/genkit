# Genkit Angular Streaming Sample

An Angular application for generating RPG characters and interactive chatbot experiences using Google's Gemini API.

## Overview

This sample application demonstrates how to implement streaming AI responses in an Angular application using Genkit and Google's Gemini API. It features two main components:
- An RPG character generator that creates and streams character profiles
- An interactive chatbot application for AI conversations

## Prerequisites

- Node.js v20+ and npm installed
- A Google Developer Gemini API key

## Setup Instructions

1. Clone the repository and install dependencies:

```bash
npm i
npm run setup
```

2. Get a Google Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to create your API key
   - Set the environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Running the Application

Start the development server:

```bash
npm run genkit:dev
```

Then access the application at [http://localhost:4200/](http://localhost:4200/)

## Project Structure

The application is built with:
- Angular for the frontend framework
- Genkit for AI integration
- Gemini 2.5 Flash for the AI model