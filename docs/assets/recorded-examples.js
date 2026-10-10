/* Recorded model replies; no model or network request on this page.
   Recorded 10 October 2026 from the released Udgam-384M-Chat-Q8_0.gguf, temperature 0.1, thinking off:
   01-03 and 06 through Ollama (default Modelfile), 04-05 through llama-server, using the scripts in examples/.
   Tools offered for 01-03: set_alarm, send_message, get_weather. Example 06 had no tools. */
window.UDGAM_RECORDED = {
  examples: [
    {
      name: 'An app action',
      job: 'App action',
      text: 'Wake me up at 6:45 tomorrow, label it gym.',
      outcome: 'Calls the right tool',
      success: true,
      explanation: 'The model picks set_alarm from the three offered tools and fills both arguments in the format the schema asks for.',
      decision: 'Call a tool',
      fields: [['Tool', 'set_alarm'], ['time', '06:45'], ['label', 'Gym']],
      application: 'Your app checks the call, sets the alarm and sends the result back. The model then replies: “Set an alarm for 6:45 am tomorrow.”',
      raw: { tool_calls: [{ function: { name: 'set_alarm', arguments: { label: 'Gym', time: '06:45' } } }] }
    },
    {
      name: 'A missing detail',
      job: 'Relevance',
      text: 'Send a message to Rahul.',
      outcome: 'Asks instead of guessing',
      success: true,
      explanation: 'send_message needs both a contact and the text. The text is missing, so the model asks for it rather than inventing a message.',
      decision: 'Ask a question',
      fields: [['Reply', 'What should the message to Rahul say?']],
      application: 'Your app shows the question and sends the person’s answer back as the next turn.',
      raw: { content: 'What should the message to Rahul say?' }
    },
    {
      name: 'No tool fits',
      job: 'Relevance',
      text: 'Book me a table for two at 8pm.',
      outcome: 'Declines and offers an alternative',
      success: true,
      explanation: 'None of the offered tools can book a restaurant. The model says so instead of forcing the request into the wrong tool.',
      decision: 'Decline',
      fields: [['Reply', 'I can’t book restaurant tables here. Want me to set an alarm for 8 pm so you can ring them?']],
      application: 'Nothing runs. Your app shows the reply.',
      raw: { content: "I can't book restaurant tables here. Want me to set an alarm for 8 pm so you can ring them?" }
    },
    {
      name: 'Document to JSON',
      job: 'Document → JSON',
      text: 'RECEIPT · Corner Cafe, 12 High St · 14/08/2027 10:42 · 2x Latte 6.40 · Blueberry muffin 2.50 · TOTAL GBP 8.90 · Paid by card',
      outcome: 'Valid against the schema',
      success: true,
      explanation: 'Given an expense schema in the system message, the model returns JSON only: ISO date, currency code, line items and total.',
      decision: 'Return JSON',
      fields: [['vendor', 'Corner Cafe'], ['date', '2027-08-14'], ['currency', 'GBP'], ['items', 'Latte ×2 6.40 · Blueberry muffin ×1 2.50'], ['total', '8.9'], ['payment_method', 'card']],
      application: 'Your app validates the JSON against the schema before saving it. Schema-valid output is not guaranteed.',
      raw: { vendor: 'Corner Cafe', date: '2027-08-14', currency: 'GBP', items: [{ name: 'Latte', quantity: 2, amount: 6.4 }, { name: 'Blueberry muffin', quantity: 1, amount: 2.5 }], total: 8.9, payment_method: 'card' }
    },
    {
      name: 'A shell command',
      job: 'Natural language → shell',
      text: 'show the 5 biggest files in my Downloads folder',
      outcome: 'Proposes one command for review',
      success: true,
      explanation: 'With the run_shell_command tool and a Mac in the system message, the model proposes one macOS command with an explanation.',
      decision: 'Propose a command',
      fields: [['command', "find $HOME/Downloads -type f -exec stat -f '%z %N' {} + | sort -rn | head -n 5"], ['explanation', 'Shows the five largest files in Downloads by size.'], ['requires_confirmation', 'false'], ['platform', 'macos']],
      application: 'Your app shows the command and runs it only after the person confirms. Never auto-execute.',
      raw: { tool_calls: [{ function: { name: 'run_shell_command', arguments: { command: "find $HOME/Downloads -type f -exec stat -f '%z %N' {} + | sort -rn | head -n 5", explanation: 'Shows the five largest files in Downloads by size.', requires_confirmation: false, platform: 'macos' } } }] }
    },
    {
      name: 'A real mistake',
      job: 'Open chat',
      text: 'What’s 15% of 2,400?',
      outcome: 'Wrong answer',
      success: false,
      explanation: 'The correct answer is 360. The model sets up a percentage formula the wrong way round and answers 16. At 384M parameters it is fluent but often wrong on arithmetic and facts.',
      decision: 'Reply in words',
      fields: [['Final answer', '16 (wrong; the answer is 360)']],
      application: 'Do not rely on it for facts or arithmetic. Compute in your app and use the model to call the function that does.',
      raw: { content: 'To determine what 15% of 2,400 is, we can use the standard formula for calculating percentages: … Final Answer: \\boxed{16}' }
    }
  ]
};
